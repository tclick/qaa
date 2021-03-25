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
from numpy.typing import ArrayLike

from ..datafile import TOPWW
from ..datafile import TRJWW
from qaa.libs import kabsch


class TestCase:
    @pytest.fixture
    def mobile(self) -> ArrayLike:
        universe = mda.Universe(TOPWW, TRJWW, in_memory=True)
        sel = universe.select_atoms("protein and name CA")
        return universe.trajectory.coordinate_array[:, sel.indices]

    @pytest.fixture
    def centered(self, mobile: ArrayLike) -> ArrayLike:
        return mobile - mobile.mean(axis=1)[:, np.newaxis]

    @pytest.fixture
    def reference(self, mobile: ArrayLike) -> ArrayLike:
        reference = mobile.mean(axis=0)
        reference -= reference.mean(axis=0)
        return reference

    def test_kabsch_fit(self, centered: ArrayLike, reference: ArrayLike):
        """
        GIVEN both mobile and reference arrays
        WHEN the mobile array is fit to the reference
        THEN a rotation square matrix is calculated with shape (3, 3)
        """
        alignment = kabsch.Kabsch()

        fit = alignment.fit(centered[0], reference)
        assert isinstance(fit, kabsch.Kabsch)
        assert fit.rotation_.shape == (3, 3)

    def test_kabsch_transform(self, centered: ArrayLike, reference: ArrayLike):
        """
        GIVEN both mobile and reference arrays
        WHEN the mobile array is fit to the reference
        THEN an aligned array is returned
        """
        pos = centered[0]
        alignment = kabsch.Kabsch()

        aligned = alignment.fit(pos, reference).transform(pos)
        assert aligned.shape == pos.shape
        assert alignment.error <= 1.0

    def test_fit_transform(self, centered: ArrayLike, reference: ArrayLike):
        """
        GIVEN both mobile and reference arrays
        WHEN the mobile array is fit to the reference
        THEN an aligned array is returned
        """
        pos = centered[0]
        alignment = kabsch.Kabsch()

        aligned = alignment.fit_transform(pos, reference)
        assert aligned.shape == pos.shape
        assert alignment.error <= 1.0
        assert alignment.rotation_.shape == (3, 3)
