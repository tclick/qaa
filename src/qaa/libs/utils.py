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
import itertools
from pathlib import Path
from typing import NoReturn

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns

from .typing import ArrayType, AtomType, PathLike, UniverseType


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
    array_like
        The coordinates with shape (n_frames, n_atoms, 3)
    """
    universe: UniverseType = mda.Universe(topology, trajectory)
    atoms: AtomType = universe.select_atoms(mask)

    positions: ArrayType = np.asarray(
        [atoms.positions for _ in universe.trajectory],
        dtype=atoms.positions.dtype,
    )
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
    return positions.reshape((n_frames, n_atoms * n_dims))


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


def save_fig(
    data: ArrayType, /, *, filename: PathLike, data_type: str = "ica", dpi: int = 600
) -> NoReturn:
    """Save projection data in a graphical form

    Parameters
    ----------
    data : array_like
        Projection data
    filename : :class:`pathlib.Path` or str
        Image file
    data_type : {'ica', 'pca'}
        ICA or PCA data
    dpi : int, default=600
        Figure resolution
    """
    fig = plt.figure(figsize=plt.figaspect(1.0))
    label = data_type.upper()[:2]
    for i, (x, y) in enumerate(itertools.combinations(range(3), 2), 1):
        ax = fig.add_subplot(2, 2, i)
        sns.scatterplot(x=data[:, x], y=data[:, y], ax=ax, marker=".")
        ax.set_xlabel(f"${label}_{x + 1:d}$")
        ax.set_ylabel(f"${label}_{y + 1:d}$")

    # Plot first 3 PCs
    ax = fig.add_subplot(2, 2, i + 1, projection="3d", proj_type="ortho")
    ax.scatter3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        marker=".",
        alpha=0.5,
    )
    ax.view_init(azim=120)
    ax.set_xlabel(f"${label}_1$")
    ax.set_ylabel(f"${label}_2$")
    ax.set_zlabel(f"${label}_3$")
    fig.suptitle(f"{data_type.upper()}")

    with Path(filename).open(mode="w") as w:
        fig.savefig(w, dpi=dpi)
