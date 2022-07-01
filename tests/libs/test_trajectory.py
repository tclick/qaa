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
"""Test the Trajectory class."""
from pathlib import Path

import numpy as np
import pytest
from datafile import TOPWW, TRJWW
from numpy import testing

from qaa.libs.trajectory import Trajectory


class TestTrajectory:
    """Test trajectory module."""

    @pytest.fixture
    def trajectory(self) -> Trajectory:
        """Return a Trajectory object.

        Returns
        -------
        Trajectory
            a trajectory object
        """
        return Trajectory(TOPWW, TRJWW, start_res=1, end_res=133)

    def test_positions(self, trajectory: Trajectory, tmp_path: Path) -> None:
        """Test output of Trajectory.get_positions().

        GIVEN A trajectory
        WHEN the method `get_positions` is called
        THEN return a 2D array of shape (n_dims, n_atoms*3)

        Parameters
        ----------
        trajectory: Trajectory
            trajectory object
        tmp_path: Path
            temporary directory
        """
        n_atoms = trajectory._universe.select_atoms(trajectory._select).n_atoms
        n_frames = trajectory._universe.trajectory.n_frames
        data_file = tmp_path / "coordinates.npy"
        positions = trajectory.get_positions(data_file)
        arr = np.memmap(data_file, dtype=np.float_, mode="r")

        assert positions.ndim == 2
        assert positions.shape == (n_frames, n_atoms * 3)
        assert data_file.exists()
        testing.assert_allclose(
            arr.reshape(n_frames, n_atoms * 3),
            positions,
            err_msg="The memmap arrays don't match.",
        )

    def test_dihedrals(self, trajectory: Trajectory, tmp_path: Path) -> None:
        """Test output of Trajectory.get_dihedrals().

        GIVEN A trajectory
        WHEN the method `get_dihedrals` is called
        THEN return a 2D array of shape (n_dims, (n_atoms-1)*4)

        Parameters
        ----------
        trajectory: Trajectory
            trajectory object
        tmp_path: Path
            temporary directory
        """
        selection = trajectory._universe.select_atoms(trajectory._select)
        n_residues = selection.residues.n_residues
        n_frames = trajectory._universe.trajectory.n_frames
        data_file = tmp_path / "coordinates.npy"
        dihedrals = trajectory.get_dihedrals(data_file)
        arr = np.memmap(data_file, dtype=np.float_, mode="r")

        assert dihedrals.ndim == 2
        assert dihedrals.shape == (n_frames, n_residues * 4)
        assert data_file.exists()
        testing.assert_allclose(
            arr.reshape(n_frames, n_residues * 4),
            dihedrals,
            err_msg="The memmap arrays don't mach.",
        )
