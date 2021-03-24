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
import MDAnalysis as mda

from .. import _MASK, create_logging_dict
from ..libs.typing import AtomType, PathLike, UniverseType


@click.command("reduce", short_help="Reduce trajectory and topology to selected atoms")
@click.option(
    "-s",
    "--top",
    "topology",
    metavar="FILE",
    default=Path.cwd().joinpath("input.top"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    help="Topology",
)
@click.option(
    "-f",
    "--traj",
    "trajectory",
    metavar="FILE",
    default=Path.cwd().joinpath("input.nc"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, resolve_path=True),
    help="Trajectory",
)
@click.option(
    "-o",
    "--outfile",
    metavar="FILE",
    default=Path.cwd().joinpath("reduced.nc"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
    help="Aligned trajectory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd().joinpath("reduce.log"),
    type=click.Path(exists=False, file_okay=True, resolve_path=True),
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
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike,
    trajectory: PathLike,
    outfile: PathLike,
    logfile: PathLike,
    start: int,
    stop: int,
    step: int,
    mask: str,
    verbose: bool,
):
    """Perform principal component analysis on the selected atoms"""
    start_time: float = time.perf_counter()

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    step: Optional[int] = step if step > 0 else None
    if start > stop != -1:
        logger.exception(
            "Final frame must be greater than start frame %d <= %d", stop, start
        )
        sys.exit(1)

    logger.info("Loading %s and %s", topology, trajectory)
    universe: UniverseType = mda.Universe(topology, trajectory)
    atoms: AtomType = universe.select_atoms(_MASK[mask.lower()])

    outfile = Path(outfile)
    logger.info("Saving aligned trajectory to %s}", outfile.as_posix())
    format: str = "NCDF" if outfile.suffix == ".nc" else outfile.suffix[1:].upper()

    with mda.Writer(outfile.as_posix(), n_atoms=atoms.n_atoms, format=format) as w:
        for ts in universe.trajectory[start:stop:step]:
            w.write(atoms)

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")
