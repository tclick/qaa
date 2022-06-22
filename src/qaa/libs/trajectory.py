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

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
from MDAnalysis.analysis import dihedrals

from .. import PathLike

logger: logging.Logger = logging.getLogger(__name__)


class Trajectory:
    def __init__(
        self,
        topology: PathLike,
        trajectory: PathLike,
        skip: int = 1,
        mask: str = "protein and name CA",
        reslist: str = "1:10",
    ):
        """Molecular dynamics (MD) trajectory

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
        reslist : str
            range of residues for calculations
        """
        self._universe: mda.Universe = mda.Universe(topology, trajectory)
        self._mask: str = f"{mask} and resnum {reslist.replace('-', ':')}"
        self._selection: mda.AtomGroup = self._universe.select_atoms(self._mask)
        self._skip: int = skip

    def get_positions(self) -> npt.NDArray[np.float_]:
        """Return a 2D matrix with shape (n_frames, 3 :math:`\times` n_atoms)

        Returns
        -------
        NDArray
            2D array of coordinates
        """
        positions = np.array(
            [
                self._selection.positions
                for _ in self._universe.trajectory[:: self._skip]
            ]
        )
        n_frames, n_dims, n_atoms = positions.shape
        positions = positions.reshape((n_frames, n_dims * n_atoms))

        return positions

    def get_dihedrals(self) -> npt.NDArray[np.float_]:
        """Return a 2D matrix with shape (n_frames, 4 :math:`\times` n_atoms)

        The backbone dihedral angles are calculated and then transformed into their
        trigonometric parts (:math:`\sin` and :math:`\cos`).

        Returns
        -------
        NDArray
            2D array of :math:`\phi`/:math:`\psi` dihedrals
        """
        try:
            phi = self._selection.residues.phi_selections()
            phi_angles = dihedrals.Dihedral(phi).run(step=self._skip).results["angles"]
            phi_angles = np.deg2rad(phi_angles)
            n_frames, n_residues = phi_angles.shape
        except AttributeError:
            logging.error("A phi angle does not exist. Please check residue selection.")

        try:
            psi = self._selection.residues.psi_selections()
            psi_angles = dihedrals.Dihedral(psi).run(step=self._skip).results["angles"]
            psi_angles = np.deg2rad(psi_angles)
        except AttributeError:
            logging.error("A psi angle does not exist. Please check residue selection.")

        angles = np.empty((n_frames, 4 * n_residues), dtype=np.float_)
        angles[:, 0::4] = np.sin(phi_angles)
        angles[:, 1::4] = np.cos(phi_angles)
        angles[:, 2::4] = np.sin(psi_angles)
        angles[:, 3::4] = np.cos(psi_angles)

        return angles
