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
"""Test kabsch module."""
from typing import Any

import mdtraj as md
import numpy as np
import pytest
from nptyping import Float
from nptyping import NDArray
from qaa.libs import kabsch

from ..datafile import TOPWW
from ..datafile import TRJWW


class TestKabsch:
    """Test Kabsch class."""

    @pytest.fixture
    def mobile(self) -> NDArray[(Any, ...), Float]:
        """Load coordinates from a trajectory file.

        Returns
        -------
        NDArray
            Trajectory
        """
        top = md.load_topology(TOPWW)
        sel = top.select("protein and name CA")
        traj = md.load(TRJWW, top=top, atom_indices=sel)
        return traj.xyz

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

    def test_kabsch_fit(
        self,
        centered: NDArray[(Any, ...), Float],
        reference: NDArray[(Any, ...), Float],
    ) -> None:
        """Test Kabsch fit method.

        GIVEN both mobile and reference arrays
        WHEN the mobile array is fit to the reference
        THEN a rotation square matrix is calculated with shape (3, 3)

        Parameters
        ----------
        centered : NDArray
            Centered coordinates
        reference : NDArray
            Reference structure
        """
        alignment = kabsch.Kabsch()

        fit = alignment.fit(centered[0], reference)
        assert isinstance(fit, kabsch.Kabsch)
        assert fit.rotation_.shape == (3, 3)

    def test_kabsch_transform(
        self,
        centered: NDArray[(Any, ...), Float],
        reference: NDArray[(Any, ...), Float],
    ) -> None:
        """Test Kabsch transform method.

        GIVEN both mobile and reference arrays
        WHEN the mobile array is fit to the reference
        THEN an aligned array is returned

        Parameters
        ----------
        centered : NDArray
            Centered coordinates
        reference : NDArray
            Reference structure
        """
        pos = centered[0]
        alignment = kabsch.Kabsch()

        aligned = alignment.fit(pos, reference).transform(pos)
        assert aligned.shape == pos.shape
        assert alignment.error <= 1.0

    def test_fit_transform(
        self,
        centered: NDArray[(Any, ...), Float],
        reference: NDArray[(Any, ...), Float],
    ) -> None:
        """Test Kabsch fit_transform method.

        GIVEN both mobile and reference arrays
        WHEN the mobile array is fit to the reference
        THEN an aligned array is returned

        Parameters
        ----------
        centered : NDArray
            Centered coordinates
        reference : NDArray
            Reference structure
        """
        pos = centered[0]
        alignment = kabsch.Kabsch()

        aligned = alignment.fit_transform(pos, reference)
        assert aligned.shape == pos.shape
        assert alignment.error <= 1.0
        assert alignment.rotation_.shape == (3, 3)
