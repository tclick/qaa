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
import time
from pathlib import Path

import click
import numpy as np
import MDAnalysis as mda

from .. import _MASK, create_logging_dict
from ..libs.align import align_to_average
from ..libs.typing import ArrayType, PathLike, UniverseType, AtomType

@click.command("align", short_help="Align trajectory to a reference")
@click.option(
    "-p",
    "--top",
    metavar="FILE",
    default=Path.cwd().joinpath("input.top"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Topology",
)
@click.option(
    "-f",
    "--traj",
    metavar="FILE",
    default=Path.cwd().joinpath("input.nc"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Trajectory",
)
@click.option(
    "-r",
    "--ref",
    metavar="FILE",
    default=Path.cwd().joinpath("average.pdb"),
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Structure file",
)
@click.option(
    "-o",
    "--outfile",
    metavar="FILE",
    default=Path.cwd().joinpath("aligned.nc"),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Aligned trajectory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="LOG",
    show_default=True,
    default=Path.cwd() / "align_traj.log",
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
    help="Starting trajectory frame",
)
@click.option(
    "-e",
    "stop",
    metavar="STOP",
    default=-1,
    show_default=True,
    type=click.IntRange(min=0, clamp=True),
    help="Final trajectory frame",
)
@click.option(
    "--dt",
    "step",
    metavar="OFFSET",
    default=1,
    show_default=True,
    type=click.IntRange(min=1, clamp=True),
    help="Trajectory output offset (0 = last frame)",
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
    "--tol",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.001,
    show_default=True,
    help="Error tolerance",
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    top: PathLike,
    traj: PathLike,
    ref: PathLike,
    outfile: PathLike,
    logfile: PathLike,
    start: int,
    stop: int,
    step: int,
    mask: str,
    tol: float,
    verbose: bool,
):
    """Align a trajectory to average structure using Kabsch fitting"""
    start_time: float = time.perf_counter()

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    if 0 < stop < start:
        msg: str = f"Final frame must be greater than start frame ({stop} <= {start}"
        logger.exception("Final frame must be greater than start frame %d <= %d", stop, start)
        raise ValueError(msg)
    frame_indices: ArrayType = np.arange(start, stop, step) if stop > 0 else None

    logger.info("Loading %s and %s", top, traj)
    universe: UniverseType = mda.Universe(top, traj)
    atoms: AtomType = universe.select_atoms(_MASK[mask.lower()])
    positions: ArrayType = atoms.positions

    logger.info("Loading reference positions")
    reference: UniverseType = mda.Universe(top, ref)
    ref_pos: ArrayType = reference.select_atoms(_MASK[mask.lower()]).positions

    logger.info("Aligning trajectory to average structures")
    aligned: ArrayType = align_to_average(positions, ref_pos, tol=tol, verbose=verbose)

    logger.info("Saving aligned trajectory to %s}", outfile)
    with mda.Writer(outfile, n_atoms=atoms.n_atoms) as w:
        for i, ts in enumerate(universe.trajectory):
            atoms.positions[:] = aligned[i]
            w.write(atoms)
