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
import logging
import logging.config
import sys
import time
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from .. import _MASK, create_logging_dict
from ..libs.typing import ArrayType, PathLike
from ..libs.utils import get_positions, reshape_positions, save_fig


@click.command("align", short_help="Align trajectory to a reference")
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
    default=Path.cwd().joinpath("input.nc"),
    show_default=True,
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
    "-b",
    "start",
    metavar="START",
    default=0,
    show_default=True,
    type=click.IntRange(min=1, clamp=True),
    help="Starting trajectory frame (0 = first frame)",
)
@click.option(
    "-e",
    "stop",
    metavar="STOP",
    default=-1,
    show_default=True,
    type=click.IntRange(min=-1, clamp=True),
    help="Final trajectory frame (-1 = final frame)",
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
    type=click.IntRange(min=0),
    help="Number of eigenmodes (-1 = all components)",
)
@click.option("-w", "--whiten", is_flag=True, help="Whitens the data")
@click.option("--bias / --no-bias", help="Calculate with population bias")
@click.option("--image", is_flag=True, help="Save graph of rmsf10 for C-alpha")
@click.option(
    "--dpi",
    default=600,
    type=click.IntRange(min=100, clamp=True),
    help="Resolution of the figure",
)
@click.option(
    "--type",
    "image_type",
    default="png",
    show_default=True,
    type=click.Choice("png pdf svg jpg".split()),
    help="Image type",
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike,
    trajectory: PathLike,
    outdir: PathLike,
    logfile: PathLike,
    start: int,
    stop: int,
    step: int,
    mask: str,
    n_modes: int,
    whiten: bool,
    bias: bool,
    image: bool,
    dpi: float,
    image_type: str,
    verbose: bool,
):
    """Align a trajectory to average structure using Kabsch fitting"""
    start_time: float = time.perf_counter()

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    outdir = Path(outdir)
    step: Optional[int] = step if step > 0 else None
    if start > stop != -1:
        logger.error(
            "Final frame must be greater than start frame %d <= %d", stop, start
        )
        sys.exit(1)

    # Extract positions and reshape to (n_frames, n_points * 3)
    logger.info("Loading trajectory positions")
    positions: ArrayType = get_positions(topology, trajectory, mask=_MASK[mask])
    positions = reshape_positions(positions[start:stop:step])
    n_samples, n_features = positions.shape
    n_components: int = n_modes if n_modes > 0 else min(n_samples, n_features)

    logger.info("Calculating PCA")
    logger.warn("Depending upon the size of the trajectory, this could take a while.")
    pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    projection: ArrayType = pca.fit_transform(positions)
    if bias and whiten:
        projection *= np.sqrt(n_samples / (n_samples - 1))
        pca.explained_variance_ *= (n_samples - 1) / n_samples
    ratio = pca.explained_variance_.cumsum() / pca.explained_variance_.sum()

    # Save data
    data = dict(
        projection=projection,
        components=pca.components_.T,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=ratio,
        singular=pca.singular_values_,
    )
    for key, value in data.items():
        with outdir.joinpath(f"{key}.csv").open(mode="w") as w:
            logger.info("Saving %s to %s", key, w.name)
            np.savetxt(w, value, delimiter=",", fmt="%.6f")

    if image:
        # Plot explained variance ratio
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 1, 1)
        sns.lineplot(x=np.arange(ratio.size) + 1, y=ratio, markers=".", ax=ax)
        ax.set_xlabel("Mode")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_xlim(left=1.0, right=n_modes)
        ax.set_ylim(bottom=0.0, top=1.1)

        fig.suptitle("Explained Variance Ratio from PCA")
        filename = outdir.joinpath("explained_variance_ratio")
        with filename.with_suffix(f".{image_type}").open(mode="w") as w:
            fig.savefig(w, dpi=dpi)

        # Plot 2D plots of PCAs
        logger.info("Plotting the PCA")
        filename = outdir.joinpath("pca").with_suffix(f".{image_type}")
        save_fig(projection, filename=filename, data_type="pca", dpi=dpi)

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")
