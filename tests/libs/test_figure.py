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
"""Test figure module."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy import random
from numpy.typing import ArrayLike
from qaa.libs import figure


class TestFigure:
    @pytest.fixture
    def data(self) -> ArrayLike:
        rng = random.default_rng()
        return rng.standard_normal((1000, 50))

    @pytest.fixture
    def fig(self) -> figure.Figure:
        return figure.Figure(method="ica")

    def test_draw(self, data: ArrayLike, fig: figure.Figure):
        """Test draw method.

        GIVEN a 2D array with shape (n_samples, n_components)
        WHEN the draw method of a `Figure` object is called
        THEN a figure with subplots is created
        """
        fig.draw(data)

        assert isinstance(fig.figure, plt.Figure)
        assert isinstance(fig.axes, plt.Axes)

    def test_save(self, data: ArrayLike, fig: figure.Figure, tmp_path: Path, mocker):
        """Test save method.

        GIVEN a filename
        WHEN the save method method of a `Figure` object is called
        THEN an image is written to disk
        """
        filename: Path = tmp_path.joinpath("test.png")
        patch = mocker.patch("matplotlib.figure.Figure.savefig")
        fig.draw(data)
        fig.save(filename)
        patch.assert_called_once()
