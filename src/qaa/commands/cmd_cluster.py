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
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Tuple

import click
import mdtraj as md
import numpy as np
import pandas as pd
from nptyping import Float
from nptyping import NDArray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .. import create_logging_dict
from .. import PathLike


@click.command("cluster", short_help="Plot data from QAA.")
@click.option(
    "-s",
    "--top",
    "topology",
    metavar="FILE",
    default=Path.cwd().joinpath("input.top"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
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
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
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
    "--outdir",
    metavar="DIR",
    default=Path.cwd(),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
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
    "--axes",
    metavar="AXES",
    nargs=3,
    default=(0, 1, 2),
    show_default=True,
    type=click.IntRange(min=0, clamp=True),
    help="Components",
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
    "--save", is_flag=True, help="Save structures from corresponding cluster center"
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike,
    trajectory: Sequence[str],
    infile: PathLike,
    outdir: PathLike,
    logfile: PathLike,
    axes: Tuple[int, int, int],
    method: bool,
    cluster: bool,
    max_iter: int,
    tol: float,
    n_clusters: int,
    n_points: int,
    save: bool,
    verbose: bool,
) -> None:
    """Perform cluster analysis on the provided data."""
    start_time: float = time.perf_counter()
    in_file = Path(infile)
    out_dir = Path(outdir)

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    data_method = "ica" if method else "pca"
    sorted_axes = np.sort(axes)
    features = [f"{data_method[:2].upper()}{_+1:d}" for _ in sorted_axes]

    # Load data
    data: pd.DataFrame
    try:
        try:
            data = pd.DataFrame(np.load(in_file)[:, axes], columns=features)
            data.index.name = "Frame"
        except ValueError:
            data = pd.read_csv(in_file, header=0, index_col=0)[features]
    except BaseException:
        raise

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
    labels = pd.Series(clustering.fit_predict(data), name="Cluster")
    centroids = pd.DataFrame(
        clustering.means_ if cluster else clustering.cluster_centers_, columns=features
    )
    centroids.index.name = "Cluster"

    data = pd.concat([labels, data.reset_index()], axis=1)

    # Prepare dataframe
    with out_dir.joinpath(f"{data_method}-cluster.csv").open(mode="w") as w:
        logger.info("Saving cluster data to %s", w.name)
        data.to_csv(w, index=False, float_format="%.6f")

    with out_dir.joinpath(f"{data_method}-cluster.npy").open(mode="wb") as wb:
        logger.info("Saving cluster data to %s", wb.name)
        np.save(wb, data.drop(["Cluster", "Frame"], axis=1))
    with out_dir.joinpath(f"{data_method}-labels.npy").open("wb") as wb:
        logger.info("Saving label data to %s", wb.name)
        np.save(wb, data["Cluster"])

    with out_dir.joinpath(f"{data_method}-centroids.csv").open("w") as w:
        logger.info("Saving centroids to %s", w.name)
        centroids.reset_index().to_csv(w, float_format="%.6f", index=False)
    with out_dir.joinpath(f"{data_method}-centroids.npy").open("wb") as wb:
        logger.info("Saving centroids to %s", wb.name)
        np.save(wb, centroids)

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

        for i, center in enumerate(centroids.iterrows()):
            idx: int = find_closest_point(center[1], data[features])
            file_no: int = int(np.searchsorted(frames, idx))
            traj: md.Trajectory = md.load_frame(filenames[file_no], idx, top=topology)

            filename = out_dir.joinpath(f"{data_method}-cluster{i}_frame{idx}.pdb")
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
    """Locate a point in the `data` closest to the `point`.

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
    distance: NDArray[(Any, ...), Float] = np.linalg.norm(data - point, axis=1)
    return int(np.where(distance == distance.min())[0])
