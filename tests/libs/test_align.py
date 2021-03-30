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
import mdtraj as md
import numpy as np
import pytest
from numpy import testing
from numpy.typing import ArrayLike
from qaa.libs import align

from ..datafile import TOPWW
from ..datafile import TRJWW


class TestAlign:
    @pytest.fixture
    def mobile(self) -> ArrayLike:
        topology = md.load_topology(TOPWW)
        indices = topology.select("protein and name CA")
        universe = md.load(TRJWW, top=topology).atom_slice(indices)
        return universe.xyz

    @pytest.fixture
    def centered(self, mobile: ArrayLike) -> ArrayLike:
        return mobile - mobile.mean(axis=1)[:, np.newaxis]

    @pytest.fixture
    def reference(self, mobile: ArrayLike) -> ArrayLike:
        reference = mobile.mean(axis=0)
        reference -= reference.mean(axis=0)
        return reference

    def test_align_trajectory(self, centered: ArrayLike, reference: ArrayLike):
        """
        GIVEN a coordinate trajectory
        WHEN aligned with its average structure
        THEN an aligned trajectory
        """
        aligned = align.align_trajectory(centered, reference, verbose=True)
        assert centered.shape == aligned.shape
        testing.assert_allclose(centered, aligned, rtol=1e-1, atol=1e-1)
