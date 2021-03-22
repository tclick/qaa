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

from qaa.libs import align


class TestAlign:
    @pytest.fixture
    def mobile(self) -> ArrayLike:
        return np.random.random((200, 100, 3))

    def test_align_trajectory(self, mobile):
        """
        GIVEN a coordinate trajectory
        WHEN aligned with its average structure
        THEN an aligned trajectory
        """
        reference: ArrayLike = mobile.mean(axis=0)
        aligned = align.align_trajectory(mobile, reference)
        assert mobile.shape == aligned.shape
        testing.assert_allclose(mobile, aligned)
