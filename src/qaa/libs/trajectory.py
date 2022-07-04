# --------------------------------------------------------------------------------------
#  Copyright (C) 2020–2022 by Timothy H. Click <Timothy.Click@briarcliff.edu>
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
"""Class for molecular dynamics trajectory.

The class will access the molecular dynamics trajectory and offer access to the
coordinates or calculate the dihedral angles.
"""
import logging
from typing import List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from MDAnalysis.lib.distances import calc_dihedrals

from .. import PathLike
from .align import align_trajectory

logger: logging.Logger = logging.getLogger(__name__)


class Trajectory:
    """Class for molecular dynamics trajectories."""

    def __init__(
        self,
        topology: PathLike,
        *trajectory: Sequence[PathLike],
        skip: int = 1,
        mask: str = "protein and name CA",
        start_res: int = 1,
        end_res: int = 10,
    ):
        """Molecular dynamics (MD) trajectory.

        Parameters
        ----------
        topology : PathLike
            topology file
        trajectory : PathLike
            trajectory file
        skip : int
            number of frames to skip
        mask : str
            atom selection
        start_res : int
            first residue number of the protein
        end_res : int
            final residue number of the protein
        """
        self._universe: mda.Universe = mda.Universe(topology, trajectory)
        self._mask: str = mask
        self._select: str = f"{mask} and resnum {start_res}:{end_res}"
        self._skip: int = skip
        self._array_shape: Tuple[int, int] = ()

    def get_positions(
        self, filename: PathLike, align: bool = True
    ) -> npt.NDArray[np.float_]:
        """Return a 2D matrix with shape (n_frames, 3n_atoms).

        Parameters
        ----------
        filename: PathLike
            location for memory-mapped array
        align : bool
            Recenter the trajectory according to the Kabsch method

        Returns
        -------
        NDArray
            2D array of coordinates
        """
        selection = self._universe.select_atoms(self._select)
        n_frames = self._universe.trajectory.n_frames // self._skip
        n_atoms = selection.n_atoms
        shape = (n_frames, n_atoms, 3)

        data = np.memmap(filename, dtype=np.float_, shape=shape, mode="w+")
        for i, _ in enumerate(self._universe.trajectory[:: self._skip]):
            data[i, :] = selection.positions
        if align:
            logger.info("Aligning coordinates.")
            align_trajectory(data, data[0])

        n_frames, n_dims, n_atoms = data.shape
        data.resize(n_frames, n_dims * n_atoms)
        data.flush()
        self._array_shape = data.shape
        logger.info(f"Saved coordinates to {filename}")

        return data

    def get_dihedrals(self, filename: PathLike) -> npt.NDArray[np.float_]:
        r"""Return a 2D matrix with shape (n_frames, 4n_atoms).

        The backbone dihedral angles are calculated and then transformed into their
        trigonometric parts (:math:`\sin` and :math:`\cos`).

        Parameters
        ----------
        filename: PathLike
            location for memory-mapped array

        Returns
        -------
        NDArray
            2D array of :math:`\phi`/:math:`\psi` dihedrals
        """
        if "backbone" not in self._mask:
            logger.info(f"Changing mask from '{self._mask}' to 'backbone'")
            self._select = self._select.replace(self._mask, "backbone")

        selection = self._universe.select_atoms(self._select)
        n_frames = self._universe.trajectory.n_frames // self._skip
        n_residues = selection.residues.n_residues
        shape = (n_frames, n_residues * 4)

        data = np.memmap(filename, dtype=np.float_, shape=shape, mode="w+")

        phi: List[mda.AtomGroup] = selection.residues.phi_selections()
        psi: List[mda.AtomGroup] = selection.residues.psi_selections()

        # Calculate phi/psi angles
        for i, _ in enumerate(self._universe.trajectory[:: self._skip]):
            # Calculate phi angles
            phi_angle = np.empty(n_residues, dtype=np.float_)
            try:
                phi_pos = np.array(
                    [
                        (atom1.position, atom2.position, atom3.position, atom4.position)
                        for atom1, atom2, atom3, atom4 in phi
                    ]
                )
                calc_dihedrals(
                    phi_pos[:, 0],
                    phi_pos[:, 1],
                    phi_pos[:, 2],
                    phi_pos[:, 3],
                    result=phi_angle,
                    backend="OpenMP",
                )
            except TypeError:
                pass

            # Calculate psi angles
            psi_angle = np.empty(n_residues, dtype=np.float_)

            try:
                psi_pos = np.array(
                    [
                        (atom1.position, atom2.position, atom3.position, atom4.position)
                        for atom1, atom2, atom3, atom4 in psi
                    ]
                )
                calc_dihedrals(
                    psi_pos[:, 0],
                    psi_pos[:, 1],
                    psi_pos[:, 2],
                    psi_pos[:, 3],
                    result=psi_angle,
                    backend="OpenMP",
                )
            except TypeError:
                pass

            data[i, 0::4] = np.sin(phi_angle)
            data[i, 1::4] = np.cos(phi_angle)
            data[i, 2::4] = np.sin(psi_angle)
            data[i, 3::4] = np.cos(psi_angle)

        data.flush()
        self._array_shape = data.shape
        logger.info(f"Saved dihedrals to {filename}")

        return data
