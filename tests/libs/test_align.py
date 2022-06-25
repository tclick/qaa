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
"""Test align module."""
import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
import pytest
from numpy import testing
from qaa.libs import align

from ..datafile import TOPWW
from ..datafile import TRJWW


class TestAlign:
    """Test alignment function."""

    @pytest.fixture
    def mobile(self) -> npt.NDArray[np.float_]:
        """Load coordinates from a trajectory file.

        Returns
        -------
        NDArray
            Trajectory
        """
        universe = mda.Universe(TOPWW, TRJWW, in_memory=True)
        selection = universe.select_atoms("protein and name CA")
        positions = np.array([selection.positions for _ in universe.trajectory])
        return positions

    @pytest.fixture
    def centered(self, mobile: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Center coordinates by their mean.

        Parameters
        ----------
        mobile : NDArray
            Positions in trajectory

        Returns
        -------
        NDArray
            Centered coordinates
        """
        return mobile - mobile.mean(axis=1)[:, np.newaxis]

    @pytest.fixture
    def reference(self, mobile: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Generate the reference coordinates from the mean of the mobile coordinates.

        Parameters
        ----------
        mobile : NDArray
            Trajectory of positions

        Returns
        -------
        NDArray
            Average coordinates of trajectory
        """
        reference = mobile.mean(axis=0)
        reference -= reference.mean(axis=0)
        return reference

    def test_align_trajectory(
        self,
        centered: npt.NDArray[np.float_],
        reference: npt.NDArray[np.float_],
    ) -> None:
        """Test alignment of trajectory.

        GIVEN a coordinate trajectory
        WHEN aligned with its average structure
        THEN an aligned trajectory

        Parameters
        ----------
        centered : NDArray
            Centered coordinates
        reference : NDArray
            Reference structure
        """
        aligned = align.align_trajectory(centered, reference, verbose=True)
        assert centered.shape == aligned.shape
        testing.assert_allclose(centered, aligned, rtol=1e-1, atol=1e-1)
