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
"""Extract frames from a trajectory into a new trajectory.

A trajectory will be read, and the selected frames will be extracted into a new file for
further processing.
"""
import glob
import logging.config
import time
from pathlib import Path
from typing import List

import click
import mdtraj as md
import numpy as np

from .. import _MASK
from .. import create_logging_dict
from .. import PathLike


@click.command("extract", short_help="Extract frames from a trajectory")
@click.option(
    "-s",
    "--top",
    "topology",
    metavar="FILE",
    default=Path.cwd().joinpath("input.top").as_posix(),
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
        Path.cwd().joinpath("input.nc").as_posix(),
    ],
    show_default=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Trajectory",
)
@click.option(
    "-x",
    "--framefile",
    default=Path.cwd().joinpath("frames.csv").as_posix(),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Frames to extract",
)
@click.option(
    "-o",
    "--outfile",
    metavar="FILE",
    default=Path.cwd().joinpath("aligned.nc").as_posix(),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Aligned trajectory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd().joinpath("align_traj.log").as_posix(),
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Log file",
)
@click.option(
    "-m",
    "--mask",
    default="ca",
    show_default=True,
    type=click.Choice(_MASK.keys()),
    help="Atom selection",
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike,
    trajectory: List[str],
    framefile: PathLike,
    outfile: PathLike,
    logfile: PathLike,
    mask: str,
    verbose: bool,
) -> None:
    """Write extracted frames to a new file."""
    start_time: float = time.perf_counter()

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    top: md.Topology = md.load_topology(topology)
    selection = top.select(_MASK[mask]) if mask != "all" else None
    filenames = (
        glob.iglob(*trajectory)
        if len(trajectory) == 1 and "*" in "".join(trajectory)
        else trajectory
    )

    logger.info("Loading frame indices from %s", framefile)
    frames = np.loadtxt(framefile, delimiter=",", dtype=int)

    multiverse = [md.load(_, top=top, atom_indices=selection) for _ in filenames]
    universe = md.join(multiverse).slice(frames)

    logger.info("Saving sliced trajectory to %s", outfile)
    universe.save(outfile)

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")
