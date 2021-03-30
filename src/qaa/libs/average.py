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
"""Determine the average structure of a trajectory."""
from typing import List
from typing import Optional

import dask
import mdtraj as md
import numpy as np

from .typing import ArrayType
from .typing import PathLike


def average_structure(
    topology: PathLike, trajectory: List[PathLike], mask: str = "all"
) -> ArrayType:
    """Compute the average structure of a trajectory.

    Parameters
    ----------
    topology : PathLike
        Topology file
    trajectory : list of PathLike
        List of trajectory files
    mask : str
        Atom selection

    Returns
    -------
    ArrayType
        The average positions
    """
    n_frames: int = 0
    positions: List[dask.delayed.Delayed] = []
    indices: Optional[ArrayType] = (
        md.load_topology(topology).select(mask) if mask != "all" else None
    )

    for filename in trajectory:
        for frames in md.iterload(filename, top=topology, atom_indices=indices):
            n_frames += frames.n_frames
            coordinates = dask.delayed(sum)(frames.xyz)
            positions.append(coordinates)

    average: ArrayType = np.sum(dask.compute(*positions), axis=0) / n_frames
    return average
