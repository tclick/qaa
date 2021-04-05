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
"""Various utilities."""
import glob
from pathlib import Path
from typing import List
from typing import Optional

import mdtraj as md
import numpy as np
from mdtraj.utils import in_units_of

from .typing import ArrayType
from .typing import PathLike


def get_average_structure(
    topology: PathLike,
    trajectory: List[PathLike],
    /,
    *,
    mask: str = "all",
    stride: Optional[int] = None,
) -> md.Trajectory:
    """Compute the average structure of a trajectory.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : list of PathLike
        List of trajectory files
    mask : str
        Atom selection
    stride : int, optional
        Number of steps to read

    Returns
    -------
    Trajectory
        The average positions
    """
    n_frames: int = 0
    positions: List[ArrayType] = []
    indices: Optional[ArrayType] = (
        md.load_topology(topology).select(mask) if mask != "all" else None
    )
    if isinstance(topology, Path):
        topology: str = topology.as_posix()
    filenames = (
        glob.iglob(*trajectory)
        if len(trajectory) == 1 and "*" in "".join(trajectory)
        else trajectory
    )

    for filename in filenames:
        for frames in md.iterload(
            filename, top=topology, atom_indices=indices, stride=stride
        ):
            n_frames += frames.n_frames
            coordinates = frames.xyz.sum(axis=0)
            positions.append(coordinates)

    # MDTraj stores positions in nanometers; we convert it to Ångstroms.
    positions: ArrayType = np.asfarray(positions, dtype=coordinates.dtype)
    frames.xyz = positions.sum(axis=0) / n_frames
    frames.unitcell_angles = frames.unitcell_angles[0, :]
    frames.unitcell_lengths = frames.unitcell_lengths[0, :]
    return frames


def get_positions(
    topology: PathLike,
    trajectory: List[PathLike],
    /,
    *,
    mask: str = "all",
    stride: Optional[int] = None,
) -> ArrayType:
    """Read a molecular dynamics trajectory and retrieve the coordinates.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : list of PathLike
        Trajectory file
    mask : str
        Selection criterion for coordinates
    stride : int, optional
        Number of steps to read

    Returns
    -------
    ArrayType
        The coordinates with shape (n_frames / step, n_atoms, 3)
    """
    if isinstance(topology, Path):
        topology: str = topology.as_posix()
    top: md.Topology = md.load_topology(topology)
    selection: Optional[ArrayType] = top.select(mask) if mask != "all" else None
    filenames = (
        glob.iglob(*trajectory)
        if len(trajectory) == 1 and "*" in "".join(trajectory)
        else trajectory
    )

    # MDTraj stores positions in nanometers; we convert it to Ångstroms.
    positions: ArrayType = np.concatenate(
        [
            frames.xyz
            for filename in filenames
            for frames in md.iterload(
                filename, top=top, atom_indices=selection, stride=stride
            )
        ],
        axis=0,
    )
    if not (
        ".gro" in "".join(filenames)
        or ".xtc" in "".join(filenames)
        or ".trj" in "".join(filenames)
        or ".tng" in "".join(filenames)
    ):
        in_units_of(positions, "nanometer", "angstroms", inplace=True)
    return positions


def reshape_positions(positions: ArrayType) -> ArrayType:
    """Reshape a n * m * 3 trajectory to a n * (m * 3) 2D matrix.

    Parameters
    ----------
    positions : ArrayType
        A 3-D matrix with shape (n_frames, n_atoms, 3)

    Returns
    -------
    ArrayType
        A 2-D array with shape (n_frames, n_atoms * 3)
    """
    n_frames, n_atoms, n_dims = positions.shape
    return positions.reshape((n_frames, n_atoms * n_dims))


def rmse(mobile: ArrayType, reference: ArrayType) -> float:
    """Calculate the root-mean-square error between two arrays.

    Parameters
    ----------
    mobile : ArrayType
        coordinates
    reference : ArrayType
        coordinates

    Returns
    -------
    float
        The error difference between the two arrays
    """
    diff: ArrayType = mobile - reference

    return np.sqrt(np.sum(np.square(diff)))
