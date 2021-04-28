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
from typing import Any
from typing import List
from typing import Optional

import mdtraj as md
import numpy as np
from mdtraj.utils import in_units_of
from nptyping import Float
from nptyping import NDArray

from .. import PathLike


def get_average_structure(
    topology: PathLike,
    trajectory: List[str],
    *,
    mask: str = "all",
    stride: Optional[int] = None,
) -> md.Trajectory:
    """Compute the average structure of a trajectory.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : list of str
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
    positions_: List[NDArray[(Any, ...), Float]] = []
    indices: Optional[NDArray[(Any, ...), Float]] = (
        md.load_topology(topology).select(mask) if mask != "all" else None
    )
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
            positions_.append(coordinates)

    # MDTraj stores positions in nanometers; we convert it to Ångstroms.
    positions: NDArray[(Any, ...), Float] = np.asfarray(positions_)
    frames.xyz = positions.sum(axis=0) / n_frames
    frames.unitcell_angles = frames.unitcell_angles[0, :]
    frames.unitcell_lengths = frames.unitcell_lengths[0, :]
    return frames


def get_positions(
    topology: PathLike,
    trajectory: List[str],
    *,
    mask: str = "all",
    stride: Optional[int] = None,
) -> NDArray[(Any, ...), Float]:
    """Read a molecular dynamics trajectory and retrieve the coordinates.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : list of str
        Trajectory file
    mask : str
        Selection criterion for coordinates
    stride : int, optional
        Number of steps to read

    Returns
    -------
    NDArray
        The coordinates with shape (n_frames / step, n_atoms, 3)
    """
    top: md.Topology = md.load_topology(topology)
    selection: Optional[NDArray[(Any, ...), Float]] = (
        top.select(mask) if mask != "all" else None
    )
    filenames = (
        glob.iglob(*trajectory)
        if len(trajectory) == 1 and "*" in "".join(trajectory)
        else trajectory
    )

    # MDTraj stores positions in nanometers; we convert it to Ångstroms.
    positions: NDArray[(Any, ...), Float] = np.concatenate(
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


def reshape_positions(
    positions: NDArray[(Any, ...), Float]
) -> NDArray[(Any, ...), Float]:
    """Reshape a n * m * 3 trajectory to a n * (m * 3) 2D matrix.

    Parameters
    ----------
    positions : array_like
        A 3-D matrix with shape (n_frames, n_atoms, 3)

    Returns
    -------
    NDArray
        A 2-D array with shape (n_frames, n_atoms * 3)
    """
    n_frames, n_atoms, n_dims = positions.shape
    return positions.reshape((n_frames, n_atoms * n_dims))


def rmse(
    mobile: NDArray[(Any, ...), Float], reference: NDArray[(Any, ...), Float]
) -> float:
    """Calculate the root-mean-square error between two arrays.

    Parameters
    ----------
    mobile : array_like
        coordinates
    reference : array_like
        coordinates

    Returns
    -------
    float
        The error difference between the two arrays
    """
    return float(np.linalg.norm(mobile - reference))
