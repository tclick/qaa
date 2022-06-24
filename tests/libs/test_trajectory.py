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
import pytest
from datafile import TOPWW
from datafile import TRJWW
from qaa.libs.trajectory import Trajectory


class TestTrajectory:
    @pytest.fixture
    def trajectory(self) -> Trajectory:
        return Trajectory(TOPWW, TRJWW, start_res=2)

    def test_positions(self, trajectory: Trajectory) -> None:
        """Test output of Trajectory.get_positions().

        GIVEN A trajectory
        WHEN the method `get_positions` is called
        THEN return a 2D array of shape (n_dims, n_atoms*3)
        """
        n_atoms = trajectory._selection.n_atoms
        n_frames = trajectory._universe.trajectory.n_frames
        positions = trajectory.get_positions()
        assert positions.ndim == 2
        assert positions.shape == (n_frames, n_atoms * 3)

    def test_dihedrals(self, trajectory) -> None:
        """Test output of Trajectory.get_dihedrals().

        GIVEN A trajectory
        WHEN the method `get_dihedrals` is called
        THEN return a 2D array of shape (n_dims, (n_atoms-1)*4)
        """
        n_atoms = trajectory._selection.n_atoms
        n_frames = trajectory._universe.trajectory.n_frames
        dihedrals = trajectory.get_dihedrals()
        assert dihedrals.ndim == 2
        assert dihedrals.shape == (n_frames, n_atoms * 4)
