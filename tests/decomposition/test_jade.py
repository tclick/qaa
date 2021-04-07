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
"""Test the Jade ICA module."""
from typing import Any
from typing import Optional

import pytest
from nptyping import Float
from nptyping import NDArray
from numpy import random
from qaa.decomposition import jade
from sklearn.utils._testing import assert_array_almost_equal


class TestJade:
    """Test JadeICA class."""

    @pytest.fixture
    def matrix(self) -> NDArray[(Any, ...), Float]:
        """Mixed signal data.

        Returns
        -------
        array_like
            Matrix generated through random number generator
        """
        rng = random.default_rng()
        return rng.standard_normal((100, 10))

    @pytest.mark.parametrize(
        "n_components",
        random.randint(1, 9, 5).tolist()
        + [
            None,
        ],
    )
    def test_fit(
        self, matrix: NDArray[(Any, ...), Float], n_components: Optional[int]
    ) -> None:
        """Test Jade ICA fit method.

        GIVEN mixed signal data
        WHEN the fit method is called
        THEN the JadeICA object is returned

        Parameters
        ----------
        matrix : ArrayLike
            Randomly generated data
        n_components : int, optional
            Number of components
        """
        n_components_ = n_components if n_components is not None else matrix.shape[1]

        ica = jade.JadeICA(n_components=n_components)
        assert isinstance(ica.fit(matrix), jade.JadeICA)
        assert ica.components_.shape == (n_components_, matrix.shape[1])

    @pytest.mark.parametrize(
        "n_components",
        random.randint(1, 9, 5).tolist()
        + [
            None,
        ],
    )
    def test_transform(
        self, matrix: NDArray[(Any, ...), Float], n_components: Optional[int]
    ) -> None:
        """Test Jade ICA transform method.

        GIVEN mixed signal data
        WHEN the transform method is called
        THEN the JadeICA object is returned

        Parameters
        ----------
        matrix : ArrayLike
            Randomly generated data
        n_components : int, optional
            Number of components
        """
        n_components_ = n_components if n_components is not None else matrix.shape[1]

        ica = jade.JadeICA(n_components=n_components)
        signal = ica.fit(matrix).transform(matrix)

        assert ica.components_.shape == (n_components_, 10)
        assert signal.shape == (100, n_components_)

    @pytest.mark.parametrize(
        "n_components",
        random.randint(1, 9, 5).tolist()
        + [
            None,
        ],
    )
    def test_fit_transform(
        self, matrix: NDArray[(Any, ...), Float], n_components: Optional[int]
    ) -> None:
        """Test Jade ICA fit_transform method.

        GIVEN mixed signal data
        WHEN the fit_transform method is called
        THEN a 2D array of signals (n_samples, n_components) is returned

        Parameters
        ----------
        matrix : ArrayLike
            Randomly generated data
        n_components : int, optional
            Number of components
        """
        n_components_ = n_components if n_components is not None else matrix.shape[1]

        ica = jade.JadeICA(n_components=n_components)
        signal = ica.fit_transform(matrix)
        assert ica.components_.shape == (n_components_, 10)
        assert signal.shape == (100, n_components_)

        ica2 = jade.JadeICA(n_components=n_components)
        signal2 = ica2.fit(matrix).transform(matrix)
        assert_array_almost_equal(ica.components_, ica2.components_)
        assert_array_almost_equal(signal, signal2)

    def test_inverse_transform(self, matrix: NDArray[(Any, ...), Float]) -> None:
        """Test Jade ICA inverse_transform method.

        GIVEN an unmixed signal matrix
        WHEN the inverse_transform method is called
        THEN an exception is raised

        Parameters
        ----------
        matrix : ArrayLike
            Randomly generated data
        """
        ica = jade.JadeICA(n_components=5)
        with pytest.raises(NotImplementedError):
            ica.inverse_transform(matrix)
