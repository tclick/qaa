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
import MDAnalysis as mda

from .typing import ArrayLike, PathLike, Universe


def get_coordinates(topology: PathLike, trajectory: PathLike) -> ArrayLike:
    """Read a molecular dynamics trajectory and retrieve the coordinates.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : PathLike
        Trajectory file

    Returns
    -------
    Array
        The coordinates with shape (n_frames, n_atoms, 3)
    """
    universe: Universe = mda.Universe(topology, trajectory)
    n_atoms = universe.atoms.n_atoms
    n_frames = universe.trajectory.n_frames
    pos: ArrayLike

    if n_atoms * n_frames >= 10_000_000:
        import dask.array as da

        pos = da.from_array([universe.atoms.positions for _ in universe.trajectory])
    else:
        import numpy as np

        pos = np.asarray([universe.atoms.positions for _ in universe.trajectory])

    return pos