# --------------------------------------------------------------------------------------
#  Copyright (C) 2020–2021 by Timothy H. Click <tclick@okstate.edu>
#
#  Permission to use, copy, modify, and/or distribute this software for any purpose
#  with or without fee is hereby granted.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
#  REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
#  FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
#  INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
#  OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
#  TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
# --------------------------------------------------------------------------------------
"""Subcommand to find the principal components of a trajectory."""
import logging.config
import time
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import click
import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import opts
from nptyping import Float
from nptyping import NDArray
from sklearn.decomposition import PCA

from .. import _MASK
from .. import create_logging_dict
from .. import PathLike
from ..libs.utils import get_positions
from ..libs.utils import reshape_positions

hv.extension("matplotlib")


@click.command("pca", short_help="Perform principal component analysis on a trajectory")
@click.option(
    "-s",
    "--top",
    "topology",
    metavar="FILE",
    default=Path.cwd().joinpath("input.top"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Topology",
)
@click.option(
    "-f",
    "--traj",
    "trajectory",
    metavar="FILE",
    default=[
        Path.cwd().joinpath("input.nc"),
    ],
    show_default=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Trajectory",
)
@click.option(
    "-o",
    "--outdir",
    metavar="DIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    help="Output directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd().joinpath("align_traj.log"),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Log file",
)
@click.option(
    "--dt",
    "step",
    metavar="OFFSET",
    default=0,
    show_default=True,
    type=click.IntRange(min=0, clamp=True),
    help="Trajectory output offset (0 = single step)",
)
@click.option(
    "-m",
    "--mask",
    default="ca",
    show_default=True,
    type=click.Choice(_MASK.keys()),
    help="Atom selection",
)
@click.option(
    "-n",
    "--nmodes",
    "n_modes",
    metavar="NMODES",
    default=0,
    show_default=True,
    type=click.IntRange(min=0, clamp=True),
    help="Number of eigenmodes (0 = all components)",
)
@click.option("-w", "--whiten", is_flag=True, help="Whitens the data")
@click.option("--bias / --no-bias", help="Calculate with population bias (N vs N-1")
@click.option(
    "--dpi",
    default=600,
    show_default=True,
    type=click.IntRange(min=100, clamp=True),
    help="Resolution of the figure",
)
@click.option(
    "--it",
    "image_type",
    default="png",
    show_default=True,
    type=click.Choice("png pdf svg jpg".split(), case_sensitive=False),
    help="Image type",
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike,
    trajectory: List[str],
    outdir: PathLike,
    logfile: PathLike,
    step: int,
    mask: str,
    n_modes: int,
    whiten: bool,
    bias: bool,
    dpi: int,
    image_type: str,
    verbose: bool,
) -> None:
    """Calculate principal components for a trajectory."""
    start_time: float = time.perf_counter()

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    out_dir = Path(outdir)
    step = step if step > 0 else 1

    # Extract positions and reshape to (n_frames, n_points * 3)
    logger.info("Loading trajectory positions")
    positions: NDArray[(Any, ...), Float] = get_positions(
        topology, trajectory, mask=_MASK[mask], stride=step
    )
    positions = reshape_positions(positions)
    n_samples, n_features = positions.shape
    n_components: Optional[int] = n_modes if n_modes > 0 else None

    logger.info("Calculating PCA")
    logger.warning(
        "Depending upon the size of the trajectory, this could take a while."
    )
    pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)

    projection = pd.DataFrame(pca.fit_transform(positions))
    projection.columns = [f"PC{_+1}" for _ in range(pca.n_components_)]
    projection.index.name = "Frame"

    logger.info("%d components were calculated.", pca.n_components_)
    explained_variance = pd.Series(pca.explained_variance_, name="Explained Variance")
    explained_variance.index += 1
    explained_variance.index.name = "Component"

    singular_values = pd.Series(pca.singular_values_, name="Singular Value")
    singular_values.index = explained_variance.index.copy()
    singular_values.index.name = explained_variance.index.name

    if bias and whiten:
        conversion = n_samples / (n_samples - 1)
        projection *= np.sqrt(conversion)
        explained_variance /= conversion

    ratio: pd.Series = explained_variance.cumsum() / explained_variance.sum() * 100.0
    ratio.name = "Percentage of Explained Variance"
    for percentage in range(80, 100, 5):
        logger.info(
            "%d components cover %.1f%% of the explained variance",
            ratio.where(ratio <= float(percentage)).dropna().size,
            percentage,
        )

    if pca.n_components_ >= 50:
        for component in np.fromiter([50, 100], dtype=int):
            logger.info(
                "%d components cover %.1f%% of the explained variance",
                component,
                ratio.iloc[component],
            )

    components = pd.DataFrame(pca.components_, index=projection.columns).T
    components.index += 1
    components.index.name = "Feature"

    # Save data
    data = dict(
        projection=projection,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=ratio,
        singular=singular_values,
    )
    for key, value in data.items():
        with out_dir.joinpath(f"{key}.csv").open(mode="w") as w:
            logger.info("Saving %s to %s", key, w.name)
            value.reset_index().to_csv(w, float_format="%.6f", index=False)
        with out_dir.joinpath(f"{key}.npy").open(mode="wb") as w:  # type: ignore
            logger.info("Saving %s to %s", key, w.name)
            np.save(w, value.values)

    # Plot explained variance ratio
    filename = out_dir.joinpath("explained_variance_ratio").with_suffix(
        "." + image_type.lower()
    )
    logger.info("Saving explained variance ratio to %s", filename)
    curve = hv.Curve(ratio, "Component", "Percentage of Explained Variance")
    points = hv.Points(ratio, ["Component", "Percentage of Explained Variance"])
    overlay = curve * points
    overlay.opts(
        opts.Curve(linewidth=1.5, color="purple"),
        opts.Points(s=1.5, marker=".", color="yellow"),
    )
    hv.output(dpi=dpi)
    hv.save(
        overlay,
        filename=filename,
        backend="matplotlib",
        title="Explained Variance Percentage",
    )

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")
