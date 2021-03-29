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

import dask.array as da
import mdtraj as md
import numpy as np

from .typing import ArrayType
from .typing import PathLike


def get_positions(
    topology: PathLike, trajectory: PathLike, /, *, mask: str = "all"
) -> ArrayType:
    """Read a molecular dynamics trajectory and retrieve the coordinates.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : PathLike
        Trajectory file
    mask : str
        Selection criterion for coordinates

    Returns
    -------
    ArrayType
        The coordinates with shape (n_frames, n_atoms, 3)
    """
    top: md.Topology = md.load_topology(topology)
    selection: ArrayType = top.select(mask)
    positions: ArrayType = da.concatenate(
        [
            frame.xyz
            for filename in glob.iglob(trajectory)
            for frame in md.iterload(filename, top=top, atom_indices=selection)
        ],
        axis=0,
    )
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
