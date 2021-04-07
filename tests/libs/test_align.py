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
from typing import Any

import mdtraj as md
import numpy as np
import pytest
from nptyping import Float
from nptyping import NDArray
from numpy import testing
from qaa.libs import align

from ..datafile import TOPWW
from ..datafile import TRJWW


class TestAlign:
    """Test alignment function."""

    @pytest.fixture
    def mobile(self) -> NDArray[(Any, ...), Float]:
        """Load coordinates from a trajectory file.

        Returns
        -------
        NDArray
            Trajectory
        """
        topology = md.load_topology(TOPWW)
        indices = topology.select("protein and name CA")
        universe = md.load(TRJWW, top=topology).atom_slice(indices)
        return universe.xyz

    @pytest.fixture
    def centered(
        self, mobile: NDArray[(Any, ...), Float]
    ) -> NDArray[(Any, ...), Float]:
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
    def reference(
        self, mobile: NDArray[(Any, ...), Float]
    ) -> NDArray[(Any, ...), Float]:
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
        centered: NDArray[(Any, ...), Float],
        reference: NDArray[(Any, ...), Float],
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
