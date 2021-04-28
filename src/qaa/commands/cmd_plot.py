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
"""Subcommand to plot data."""
import logging.config
import time
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import pandas as pd

from .. import create_logging_dict
from .. import PathLike
from ..libs.figure import Figure


@click.command("cluster", short_help="Plot data from QAA.")
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
    "--label",
    metavar="FILE",
    default=Path.cwd().joinpath("labels.npy"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Cluster labels",
)
@click.option(
    "-c",
    "--centroid",
    metavar="FILE",
    default=Path.cwd().joinpath("centroids.csv"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Cluster labels",
)
@click.option(
    "-o",
    "--outfile",
    metavar="FILE",
    default=Path.cwd().joinpath("cluster.png"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Image file",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "plot.log",
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Log file",
)
@click.option(
    "--axes",
    metavar="AXES",
    nargs=3,
    default=(0, 1, 2),
    type=click.IntRange(min=0, clamp=True),
    help="Components to plot",
)
@click.option("--ica / --pca", "method", default=True, help="Type of data")
@click.option(
    "--dpi",
    metavar="DPI",
    default=600,
    show_default=True,
    type=click.IntRange(min=100, clamp=True),
    help="Resolution of the figure",
)
@click.option(
    "--azim",
    "azimuth",
    metavar="AZIMUTH",
    default=120,
    show_default=True,
    type=click.IntRange(min=0, max=359, clamp=True),
    help="Azimuth rotation of 3D plot",
)
@click.option(
    "--elev",
    "elevation",
    metavar="ELEVATION",
    default=30,
    show_default=True,
    type=click.IntRange(min=0, max=90, clamp=True),
    help="Elevation of 3D plot",
)
@click.option("--cluster", is_flag=True, help="Cluster analysis")
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    infile: PathLike,
    label: PathLike,
    centroid: PathLike,
    outfile: PathLike,
    logfile: PathLike,
    axes: Tuple[int, int, int],
    method: bool,
    dpi: int,
    azimuth: int,
    elevation: int,
    cluster: bool,
    verbose: bool,
) -> None:
    """Visualize the data."""
    start_time: float = time.perf_counter()
    in_file = Path(infile)

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    data_method = "ica" if method else "pca"
    sorted_axes = np.sort(axes)
    features = [f"{data_method[:2].upper()}{_+1:d}" for _ in sorted_axes]

    # Load data
    logger.info("Loading %s", in_file)

    index = "Frame"
    data = read_file(in_file, index=index)
    if data.empty:
        raise SystemExit(f"Unable to read {in_file}")

    data.columns = (
        [f"{data_method[:2].upper()}{_+1:d}" for _ in range(data.columns.size)]
        if np.issubdtype(data.columns, int)
        else data.columns
    )
    try:
        data = pd.concat([data["Cluster"], data[features].reset_index()], axis=1)
    except KeyError:
        data = data[features].reset_index()

    # Load labels, if exists
    centroid_data: pd.DataFrame = pd.DataFrame()
    if cluster:
        label_data = read_file(Path(label))
        if "Cluster" not in data.columns and not label_data.empty:
            label_data.columns = ["Cluster"]
            data = pd.concat([label_data, data], axis=1)

        # Load centroid data, if exists
        centroid_data = read_file(Path(centroid), index="Cluster")
        if not centroid_data.empty:
            centroid_data = centroid_data.set_index("Cluster")
            centroid_data.columns = features
            centroid_data = centroid_data.reset_index()
    else:
        n_samples, _ = data.shape
        label_data = pd.Series(np.zeros(n_samples, dtype=int), name="Cluster")
        if "Cluster" not in data.columns:
            data = pd.concat([label_data, data], axis=1)
        else:
            data["Cluster"] = label_data.copy()

    # Prepare cluster analysis
    figure = Figure(azim=azimuth, elevation=elevation)
    logger.info("Preparing figures")
    figure.draw(data, centers=centroid_data)

    logger.info("Saving visualization data to %s", outfile)
    figure.save(outfile, dpi=dpi)

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")


def read_file(filename: Path, *, index: str = "") -> pd.DataFrame:
    """Read a file and return a DataFrame.

    Parameters
    ----------
    filename : Path
        File to open
    index : str
        Index name

    Returns
    -------
    pd.DataFrame
        Parsed data
    """
    try:
        df = (
            pd.read_csv(filename, header=0)
            if filename.suffix == ".csv"
            else pd.DataFrame(np.load(filename))
        )
        if not df.index.name:
            df.index.name = index
    except Exception:
        df = pd.DataFrame()
    return df
