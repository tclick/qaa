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
import MDAnalysis as mda
import numpy as np
import pytest
import xarray as xr
from numpy import testing

from qaa.libs import utils

from ..datafile import TOPWW, TRJWW


class TestUtils:
    @pytest.fixture
    def universe(self) -> mda.Universe:
        return mda.Universe(TOPWW, TRJWW)

    @pytest.fixture
    def n_atoms(self, universe: mda.Universe) -> int:
        return universe.atoms.n_atoms

    @pytest.fixture
    def n_frames(self, universe: mda.Universe) -> int:
        return universe.trajectory.n_frames

    def test_positions_xarray(
        self, universe: mda.Universe, n_atoms: int, n_frames: int
    ):
        """
        GIVEN topology and trajectory filenames
        WHEN the get_positions function is called
        THEN return an xarray of positions with shape (n_frames, n_atoms, 3)
        """
        array = utils.get_positions(TOPWW, TRJWW)
        assert array.shape == (n_frames, n_atoms, 3)
        testing.assert_allclose(array[0], universe.atoms.positions)
        assert isinstance(array, xr.DataArray)

        universe.trajectory[-1]
        testing.assert_allclose(array[-1], universe.atoms.positions)

    def test_positions_array(self, universe: mda.Universe, n_atoms: int, n_frames: int):
        """
        GIVEN topology and trajectory filenames
        WHEN the get_positions function is called
        THEN return a array of positions with shape (n_frames, n_atoms, 3)
        """
        array = utils.get_positions(TOPWW, TRJWW, return_type="array")
        assert array.shape == (n_frames, n_atoms, 3)
        testing.assert_allclose(array[0], universe.atoms.positions)
        assert isinstance(array, np.ndarray)

        universe.trajectory[-1]
        testing.assert_allclose(array[-1], universe.atoms.positions)

    def test_reshape_array(self, universe: mda.Universe, n_atoms: int, n_frames: int):
        """
        GIVEN a coordinate matrix of shape (n_frames, n_points, 3)
        WHEN calling the reshape_position function
        THEN a 2D-array of shape (n_frames, n_points * 3) will be returned
        """
        positions = np.array([universe.atoms.positions for _ in universe.trajectory])
        new_positions = utils.reshape_positions(positions)

        assert new_positions.shape == (n_frames, n_atoms * 3)
        assert isinstance(new_positions, np.ndarray)

    def test_reshape_xarray(self, universe: mda.Universe, n_atoms: int, n_frames: int):
        """
        GIVEN a coordinate matrix of shape (n_frames, n_points, 3)
        WHEN calling the reshape_position function
        THEN a 2D-array of shape (n_frames, n_points * 3) will be returned
        """
        positions = xr.DataArray(
            [universe.atoms.positions for _ in universe.trajectory],
            dims="frame name dim".split(),
        )
        new_positions = utils.reshape_positions(positions)

        assert new_positions.shape == (n_frames, n_atoms * 3)
        assert isinstance(new_positions, xr.DataArray)

    def test_rmse(self, universe):
        """
        GIVEN the same set of coordinates twice
        WHEN rmse is called
        THEN a value of 0.0 should be returned
        """
        positions = universe.select_atoms("name CA").positions
        error = utils.rmse(positions, positions)
        testing.assert_allclose(error, 0.0)
