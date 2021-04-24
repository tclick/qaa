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
"""Test utilities module."""
import mdtraj as md
import numpy as np
import pytest
from numpy import testing
from qaa.libs import utils

from ..datafile import TOPWW
from ..datafile import TRJWW


class TestUtils:
    """Test functions in utils module."""

    @pytest.fixture
    def universe(self) -> md.Trajectory:
        """Create a trajectory.

        Returns
        -------
        Trajectory
            Molecular dynamics trajectory
        """
        return md.load(TRJWW, top=TOPWW)

    @pytest.fixture
    def n_atoms(self, universe: md.Trajectory) -> int:
        """Return number of atoms in the system.

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory

        Returns
        -------
        int
            Number of atoms
        """
        return universe.topology.n_atoms

    @pytest.fixture
    def n_frames(self, universe: md.Trajectory) -> int:
        """Return the number of frames in the trajectory.

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory

        Returns
        -------
        int
            Number of frames
        """
        return universe.n_frames

    def test_average(self, universe: md.Trajectory, n_atoms: int) -> None:
        """Test get_average_structure function.

        GIVEN topology and trajectory filenames
        WHEN the get_average_structure function is called
        THEN the average coordinates are computed

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory
        n_atoms : int
            Number of atoms
        """
        average = utils.get_average_structure(
            TOPWW,
            [
                TRJWW,
            ],
        )
        assert average.xyz.shape == (1, n_atoms, 3)
        universe_average = universe.xyz.mean(axis=0)
        testing.assert_allclose(average.xyz[0], universe_average, rtol=1e-6)

    def test_select_average(self, universe: md.Trajectory) -> None:
        """Test get_average_structure function using atom selection.

        GIVEN topology and trajectory filenames and an atom selection
        WHEN the get_average_structure function is called
        THEN the average coordinates are computed

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory
        """
        mask = "protein and name CA"
        atoms = universe.topology.select(mask)
        n_atoms = atoms.size

        average = utils.get_average_structure(
            TOPWW,
            [
                TRJWW,
            ],
            mask=mask,
        )
        assert average.xyz.shape == (1, n_atoms, 3)
        universe_average = universe.atom_slice(atoms).xyz.mean(axis=0)
        testing.assert_allclose(average.xyz[0], universe_average)

    def test_positions(
        self, universe: md.Trajectory, n_atoms: int, n_frames: int
    ) -> None:
        """Test get_positions function.

        GIVEN topology and trajectory filenames
        WHEN the get_positions function is called
        THEN return a array of positions with shape (n_frames, n_atoms, 3)

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory
        n_atoms : int
            Number of atoms
        n_frames : int
            Number of frames
        """
        array = utils.get_positions(
            TOPWW,
            [
                TRJWW,
            ],
        )
        assert array.shape == (n_frames, n_atoms, 3)
        testing.assert_allclose(array[0], universe.xyz[0] * 10)
        assert isinstance(array, np.ndarray)
        testing.assert_allclose(array[-1], universe.xyz[-1] * 10)

    def test_select_positions(self, universe: md.Trajectory, n_frames: int) -> None:
        """Test get_positions function using atom selection.

        GIVEN topology and trajectory filenames and an atom selection
        WHEN the get_positions function is called
        THEN return a array of positions with shape (n_frames, n_atoms, 3)

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory
        n_frames : int
            Number of frames
        """
        mask = "protein and name CA"
        atoms = universe.topology.select(mask)
        n_atoms = atoms.size

        array = utils.get_positions(
            TOPWW,
            [
                TRJWW,
            ],
            mask=mask,
        )
        assert array.shape == (n_frames, n_atoms, 3)
        testing.assert_allclose(array[0], universe.atom_slice(atoms).xyz[0] * 10)
        assert isinstance(array, np.ndarray)

    def test_reshape_array(
        self, universe: md.Trajectory, n_atoms: int, n_frames: int
    ) -> None:
        """Test reshape_positions function.

        GIVEN a coordinate matrix of shape (n_frames, n_points, 3)
        WHEN calling the reshape_position function
        THEN a 2D-array of shape (n_frames, n_points * 3) will be returned

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory
        n_atoms : int
            Number of atoms
        n_frames : int
            Number of frames
        """
        new_positions = utils.reshape_positions(universe.xyz)

        assert new_positions.shape == (n_frames, n_atoms * 3)
        assert isinstance(new_positions, np.ndarray)

    def test_rmse(self, universe: md.Trajectory) -> None:
        """Test rmse function.

        GIVEN the same set of coordinates twice
        WHEN rmse is called
        THEN a value of 0.0 should be returned

        Parameters
        ----------
        universe : Trajectory
            Molecular dynamics trajectory
        """
        selection = universe.top.select("protein and name CA")
        positions = universe.xyz[:, selection]
        error = utils.rmse(positions, positions)
        testing.assert_allclose(error, 0.0)
