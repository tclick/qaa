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
import logging.config
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np
from sklearn.decomposition import FastICA

from .. import _MASK
from .. import create_logging_dict
from ..decomposition.jade import JadeICA
from ..libs.figure import Figure
from ..libs.typing import ArrayType
from ..libs.typing import PathLike
from ..libs.utils import get_positions
from ..libs.utils import reshape_positions


@click.command("qaa", short_help="Perform quasi-anharmonic analysis of a trajectory")
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
    "-b",
    "start",
    metavar="START",
    default=0,
    show_default=True,
    type=click.IntRange(min=0, clamp=True),
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
    type=click.IntRange(min=0, clamp=True),
    help="Number of eigenmodes (0 = all components)",
)
@click.option("--jade / --fastica", "method", default=True, help="QAA method")
@click.option("-w", "--whiten", is_flag=True, help="Whitens the data")
@click.option(
    "--iter",
    "max_iter",
    metavar="MAXITER",
    default=200,
    show_default=True,
    type=click.IntRange(min=1, clamp=True),
    help="Maximum number of iterations for FastICA",
)
@click.option(
    "--tol",
    metavar="TOL",
    default=0.001,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, clamp=True),
    help="Maximum number of iterations for FastICA",
)
@click.option("--image", is_flag=True, help="Save images of QAA projections")
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
    method: bool,
    whiten: bool,
    max_iter: int,
    tol: float,
    image: bool,
    dpi: float,
    image_type: str,
    verbose: bool,
):
    """Perform quasi-anharmonic analysis on a trajectory"""
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
    stop: Optional[int] = stop if stop != -1 else None

    # Extract positions and reshape to (n_frames, n_points * 3)
    logger.info("Loading trajectory positions")
    positions: ArrayType = get_positions(topology, trajectory, mask=_MASK[mask])
    positions = reshape_positions(positions[start:stop:step])
    n_samples, n_features = positions.shape
    n_components: int = n_modes if n_modes > 0 else min(n_samples, n_features)

    logger.info("Calculating PCA")
    logger.warn("Depending upon the size of the trajectory, this could take a while.")

    qaa_method = "Jade" if method else "Fast"
    logger.info("Running QAA using %sICA", qaa_method)
    logger.warning(
        "Depending upon the size of your system and number of modes,"
        "this could take a while..."
    )
    qaa = (
        JadeICA(n_components=n_components, verbose=verbose)
        if method
        else FastICA(
            n_components=n_components,
            whiten=whiten,
            fun="cube",
            max_iter=max_iter,
            tol=tol,
            random_state=0,
        )
    )
    signals = qaa.fit_transform(positions)
    with outdir.joinpath("qaa-signals.csv").open(mode="w") as w:
        logger.info("Saving QAA data to %s", w.name)
        np.savetxt(w, signals, delimiter=",", fmt="%.6f")
    with outdir.joinpath("qaa-signals.npy").open(mode="wb") as w:
        logger.info("Saving QAA data to %s", w.name)
        np.save(w, signals)

    if image:
        # Plot 2D and 3D plots of ICs
        logger.info("Plotting the ICA")
        filename = outdir.joinpath("qaa").with_suffix(f".{image_type}")
        figure = Figure(method="ica")
        figure.draw(signals)
        figure.save(filename, dpi=dpi)

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")
