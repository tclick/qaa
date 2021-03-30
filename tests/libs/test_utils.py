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
    @pytest.fixture
    def universe(self) -> md.Trajectory:
        return md.load(TRJWW, top=TOPWW)

    @pytest.fixture
    def n_atoms(self, universe: md.Trajectory) -> int:
        return universe.topology.n_atoms

    @pytest.fixture
    def n_frames(self, universe: md.Trajectory) -> int:
        return universe.n_frames

    def test_average(self, universe: md.Trajectory, n_atoms: int):
        """
        GIVEN topology and trajectory filenames
        WHEN the get_average_structure function is called
        THEN the average coordinates are computed
        """
        average = utils.get_average_structure(
            TOPWW,
            [
                TRJWW,
            ],
        )
        assert average.xyz.shape == (1, n_atoms, 3)
        universe_average = universe.xyz.mean(axis=0)
        testing.assert_allclose(average.xyz[0], universe_average)

    def test_select_average(self, universe: md.Trajectory):
        """
        GIVEN topology and trajectory filenames and an atom selection
        WHEN the get_average_structure function is called
        THEN the average coordinates are computed
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

    def test_positions(self, universe: md.Trajectory, n_atoms: int, n_frames: int):
        """
        GIVEN topology and trajectory filenames
        WHEN the get_positions function is called
        THEN return a array of positions with shape (n_frames, n_atoms, 3)
        """
        array = utils.get_positions(
            TOPWW,
            [
                TRJWW,
            ],
        )
        assert array.shape == (n_frames, n_atoms, 3)
        testing.assert_allclose(array[0], universe.xyz[0])
        assert isinstance(array, np.ndarray)
        testing.assert_allclose(array[-1], universe.xyz[-1])

    def test_select_positions(self, universe: md.Trajectory, n_frames: int):
        """
        GIVEN topology and trajectory filenames and an atom selection
        WHEN the get_positions function is called
        THEN return a array of positions with shape (n_frames, n_atoms, 3)
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
        testing.assert_allclose(array[0], universe.atom_slice(atoms).xyz[0])
        assert isinstance(array, np.ndarray)

    def test_reshape_array(self, universe: md.Trajectory, n_atoms: int, n_frames: int):
        """
        GIVEN a coordinate matrix of shape (n_frames, n_points, 3)
        WHEN calling the reshape_position function
        THEN a 2D-array of shape (n_frames, n_points * 3) will be returned
        """
        new_positions = utils.reshape_positions(universe.xyz)

        assert new_positions.shape == (n_frames, n_atoms * 3)
        assert isinstance(new_positions, np.ndarray)

    def test_rmse(self, universe):
        """
        GIVEN the same set of coordinates twice
        WHEN rmse is called
        THEN a value of 0.0 should be returned
        """
        selection = universe.top.select("protein and name CA")
        positions = universe.xyz[:, selection]
        error = utils.rmse(positions, positions)
        testing.assert_allclose(error, 0.0)
