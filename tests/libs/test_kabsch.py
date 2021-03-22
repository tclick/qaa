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

import numpy as np
import pytest
from numpy import testing
from numpy.typing import ArrayLike

from qaa.libs import kabsch


class TestCase:
    @pytest.fixture
    def data(self):
        return np.random.random((100, 3))

    def test_kabsch_fit(self, data):
        """
        GIVEN an array
        WHEN aligned to itself using the Kabsch method
        THEN return the array
        """
        actual: ArrayLike = kabsch.kabsch_fit(data, data)
        testing.assert_allclose(actual, data)

    def test_kabsch(self, data):
        """
        GIVEN an array
        WHEN aligned to itself using the Kabsch method
        THEN return the array
        """
        actual: ArrayLike = kabsch.kabsch(data, data)
        testing.assert_allclose(actual, np.eye(3), atol=1e-6)

    def test_kabsch_rotate(self, data):
        """
        GIVEN an array
        WHEN rotated onto itself using the Kabsch method
        THEN return the 3x3 rotation array
        """
        actual: ArrayLike = kabsch.kabsch_rotate(data, data)
        testing.assert_allclose(actual, data)
