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
"""Test the Jade ICA module"""
import numpy as np
import pytest
from numpy import random
from numpy import typing
from sklearn.utils._testing import assert_array_almost_equal

from qaa.decomposition import jade


class TestJade:
    @pytest.fixture
    def matrix(self) -> typing.ArrayLike:
        rng = random.default_rng()
        return rng.standard_normal((100, 10))

    def test_fit_transform(self, matrix: typing.ArrayLike):
        X = matrix
        for n_components in (5, None):
            n_components_ = n_components if n_components is not None else X.shape[1]

            ica = jade.JadeICA(n_components=n_components)
            Xt = ica.fit_transform(X)
            assert ica.components_.shape == (n_components_, 10)
            assert Xt.shape == (100, n_components_)

            ica.fit(X)
            assert ica.components_.shape == (n_components_, 10)
            Xt2 = ica.transform(X)

            assert_array_almost_equal(Xt, Xt2)

    def test_inverse_transform(self, matrix):
        ica = jade.JadeICA(n_components=5)
        with pytest.raises(NotImplementedError):
            ica.inverse_transform(matrix)
