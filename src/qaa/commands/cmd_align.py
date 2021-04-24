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
"""Align trajectories to their average structure."""
import logging.config
import time
from pathlib import Path
from typing import Any
from typing import List

import click
import mdtraj as md
import numpy as np
from mdtraj.utils import in_units_of
from nptyping import Float
from nptyping import NDArray

from .. import _MASK
from .. import create_logging_dict
from .. import PathLike
from ..libs.align import align_trajectory
from ..libs.utils import get_average_structure
from ..libs.utils import get_positions


@click.command("align", short_help="Align trajectory to a reference")
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
    "-r",
    "--ref",
    "reference",
    metavar="FILE",
    default=Path.cwd().joinpath("average.pdb").as_posix(),
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Average structure of trajectory",
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
    "--tol",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.001,
    show_default=True,
    help="Error tolerance",
)
@click.option("-v", "--verbose", is_flag=True, help="Noisy output")
def cli(
    topology: PathLike,
    trajectory: List[str],
    reference: PathLike,
    outfile: PathLike,
    logfile: PathLike,
    step: int,
    mask: str,
    tol: float,
    verbose: bool,
) -> None:
    """Align a trajectory to average structure using Kabsch fitting."""
    start_time: float = time.perf_counter()

    # Setup logging
    logging.config.dictConfig(create_logging_dict(logfile))
    logger: logging.Logger = logging.getLogger(__name__)

    step = step if step > 0 else 1

    logger.info("Loading %s and %s", topology, trajectory)
    positions: NDArray[(Any, ...), Float] = get_positions(
        topology, trajectory, mask=_MASK[mask], stride=step
    )

    # Calculate average structure
    ref_traj: md.Trajectory = get_average_structure(
        topology, trajectory, mask=_MASK[mask], stride=step
    )

    logger.info("Saving average structure to %s", reference)
    ref_traj.save(reference)
    unitcell_angles: NDArray[(Any, ...), Float] = ref_traj.unitcell_angles.copy()
    unitcell_lengths: NDArray[(Any, ...), Float] = ref_traj.unitcell_lengths.copy()
    unitcell_vectors: NDArray[(Any, ...), Float] = ref_traj.unitcell_vectors.copy()
    if not (
        ".gro" in "".join(trajectory)
        or ".xtc" in "".join(trajectory)
        or ".trj" in "".join(trajectory)
        or ".tng" in "".join(trajectory)
    ):
        in_units_of(ref_traj.xyz, "nanometer", "angstroms", inplace=True)

    logger.info("Aligning trajectory to average structures")
    ref_traj.xyz = align_trajectory(
        positions, ref_traj.xyz[0], tol=tol, verbose=verbose
    )
    n_frames = ref_traj.n_frames
    ref_traj.time = np.arange(n_frames)
    ref_traj.unitcell_angles = np.repeat(unitcell_angles, n_frames, axis=0)
    ref_traj.unitcell_lengths = np.repeat(unitcell_lengths, n_frames, axis=0)
    ref_traj.unitcell_vectors = np.repeat(unitcell_vectors, n_frames, axis=0)
    if not (
        ".gro" in "".join(trajectory)
        or ".xtc" in "".join(trajectory)
        or ".trj" in "".join(trajectory)
        or ".tng" in "".join(trajectory)
    ):
        in_units_of(ref_traj.xyz, "angstroms", "nanometer", inplace=True)

    logger.info("Saving aligned trajectory to %s}", outfile)
    ref_traj.save(outfile)

    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    if verbose:
        output: str = time.strftime("%H:%M:%S", struct_time)
        logger.info(f"Total execution time: {output}")
