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
"""Cluster data into regions."""
import glob
import logging.config
import time
from os import PathLike
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Tuple

import click
import mdtraj as md
import numpy as np
from nptyping import Float
from nptyping import NDArray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .. import create_logging_dict
from ..libs.figure import Figure


@click.command("cluster", short_help="Plot data from QAA.")
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
    "-i",
    "--infile",
    metavar="FILE",
    default=Path.cwd().joinpath("input.csv"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Data file for analysis",
)
@click.option(
    "-o",
    "--outfile",
    metavar="FILE",
    default=Path.cwd().joinpath("cluster.png"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Clustered image file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "image.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "--axes", nargs=3, default=(0, 1, 2), type=click.IntRange(min=0, clamp=True)
)
@click.option("--ica / --pca", "method", default=True, help="Type of data")
@click.option("--gmm / --kmeans", "cluster", default=True, help="Clustering method")
@click.option(
    "--iter",
    "max_iter",
    metavar="MAXITER",
    default=200,
    show_default=True,
    type=click.IntRange(min=1, clamp=True),
    help="Maximum number of iterations for clustering",
)
@click.option(
    "--tol",
    metavar="TOL",
    default=0.001,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0, clamp=True),
    help="Maximum tolerance for clustering",
)
@click.option(
    "-n",
    "--nclusters",
    "n_clusters",
    default=3,
    show_default=True,
    type=click.IntRange(min=2, max=4, clamp=True),
    help="Number of cluster centers to plot",
)
@click.option(
    "--dp",
    "n_points",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, clamp=True),
    help="Number of points to skip for plotting",
)
@click.option(
    "--dpi",
    default=600,
    show_default=True,
    type=click.IntRange(min=100, clamp=True),
    help="Resolution of the figure",
)
@click.option(
    "--azim",
    default=-1.0,
    show_default=True,
    type=click.FloatRange(min=-1.0, max=359.0, clamp=True),
    help="Azimuth rotation for 3D plot",
)
@click.option(
    "--save", is_flag=True, help="Save structures from corresponding cluster center"
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike[str],
    trajectory: Sequence[str],
    infile: PathLike[str],
    outfile: PathLike[str],
    logfile: PathLike[str],
    axes: Tuple[int, int, int],
    method: bool,
    cluster: bool,
    max_iter: int,
    tol: float,
    n_clusters: int,
    n_points: int,
    dpi: int,
    azim: float,
    save: bool,
    verbose: bool,
) -> None:
    """Perform cluster analysis on the provided data."""
    start_time: float = time.perf_counter()
    in_file = Path(infile)
    out_file = Path(outfile)

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    if axes[2] <= axes[1] <= axes[0]:
        raise IndexError("Axes must be in increasing order")

    # Load data
    data: NDArray[(Any, ...), Float]
    try:
        try:
            data = np.loadtxt(in_file, delimiter=",")
        except UnicodeDecodeError:
            data = np.load(in_file)
    except BaseException:
        raise

    data_method = "ica" if method else "pca"

    # Select clustering method and cluster data
    clustering = (
        GaussianMixture(
            n_components=n_clusters,
            max_iter=max_iter,
            tol=tol,
        )
        if cluster
        else KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
        )
    )
    labels: NDArray[(Any, ...), Float] = clustering.fit_predict(data[:, axes])
    centers = clustering.means_ if cluster else clustering.cluster_centers_

    # Prepare cluster analysis
    azim = azim if 0.0 <= azim < 360.0 else 0.0
    figure = Figure(n_points=n_points, method=data_method, labels=labels, azim=azim)
    figure.draw(data[:, axes], centers=centers)

    logger.info("Saving cluster data to %s", outfile)
    figure.save(outfile, dpi=dpi)

    with out_file.with_suffix(".csv").open(mode="w") as w:
        logger.info("Saving cluster labels to %s", w.name)
        np.savetxt(w, labels, delimiter=",", fmt="%d")

    with out_file.with_suffix(".npy").open(mode="wb") as w:  # type: ignore
        logger.info("Saving cluster labels to %s", w.name)
        np.save(w, labels)

    if save:
        # Find all trajectories and determine total frames per trajectory
        filenames: Sequence[str] = glob.glob(*trajectory)
        frames: NDArray[(Any, ...), Float] = np.asarray(
            [
                sum([_.n_frames for _ in md.iterload(filename, top=topology)])
                for filename in filenames
            ],
            dtype=int,
        ).cumsum()

        for i, center in enumerate(centers):
            idx: int = find_closest_point(center, data[:, axes])
            file_no: int = int(np.searchsorted(frames, idx))
            traj: md.Trajectory = md.load_frame(filenames[file_no], idx, top=topology)

            filename = out_file.parent.joinpath(f"cluster{i}_frame{idx}.pdb")
            logger.info("Saving frame %d of cluster %d to %s", idx, i, filename)
            traj.save(filename.as_posix())

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")


def find_closest_point(
    point: NDArray[(Any, ...), Float], data: NDArray[(Any, ...), Float]
) -> int:
    """Locate a oint in the `data` closest to the `point`.

    Parameters
    ----------
    point : NDArray[(Any, ...), Float]
        Point of interest with size (n_features, )
    data : NDArray[(Any, ...), Float]
        Data to search with shape (n_samples, n_features)

    Returns
    -------
    int
        Index of value closes to `point`
    """
    distance: NDArray[(Any, ...), Float] = np.fromiter(
        [np.linalg.norm(_ - point) for _ in data], dtype=point.dtype
    )
    return int(np.where(distance == distance.min())[0])
