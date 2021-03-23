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
import dask.array as da
import MDAnalysis as mda
import numpy as np
import xarray as xr

from .typing import ArrayType, PathLike, UniverseType


def get_positions(
    topology: PathLike, trajectory: PathLike, /, *, return_type: str = "xarray"
) -> ArrayType:
    """Read a molecular dynamics trajectory and retrieve the coordinates.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : PathLike
        Trajectory file
    return_type : {"xarray", "array"}
        Positions either as a labeled array or a NumPy array

    Returns
    -------
    array_like
        The coordinates with shape (n_frames, n_atoms, 3)
    """
    universe: UniverseType = mda.Universe(topology, trajectory)
    frames = list(range(universe.trajectory.n_frames))
    names = universe.atoms.names
    dims = "x y z".split()

    positions: ArrayType = xr.DataArray(
        [universe.atoms.positions for _ in universe.trajectory],
        coords=[frames, names, dims],
        dims=["frame", "name", "dim"],
    )
    if return_type == "array":
        return positions.data
    else:
        return positions


def reshape_positions(positions: ArrayType) -> ArrayType:
    """Reshape a n * m * 3 trajectory to a n * (m * 3) 2D matrix.

    Parameters
    ----------
    positions : array_like
        A 3-D matrix with shape (n_frames, n_atoms, 3)

    Return
    ------
    array_like
        A 2-D array with shape (n_frames, n_atoms * 3)
    """
    n_frames, n_atoms, n_dims = positions.shape
    new_positions: ArrayType
    if isinstance(positions, xr.DataArray):
        frames = np.arange(n_frames) + 1
        dims = "x y z".split() * n_atoms
        new_positions = xr.DataArray(
            positions.data.reshape((n_frames, n_atoms * n_dims)),
            coords=[frames, dims],
            dims=["frame", "coords"],
        )
    else:
        new_positions = positions.reshape((n_frames, n_atoms * n_dims))

    return new_positions


def rmse(mobile: ArrayType, reference: ArrayType) -> float:
    """Calculate the root-mean-square error between two arrays

    Parameters
    ----------
    mobile : array_like
        coordinates
    reference : array_like
        coordinates

    Returns
    -------
    error : float
        The error difference between the two arrays
    """
    diff: ArrayType = mobile - reference

    return np.sqrt(np.sum(np.square(diff)))
