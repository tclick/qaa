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

import holoviews as hv
import pandas as pd
import pytest
from numpy import random
from pytest_mock import MockerFixture
from qaa.libs import figure


class TestFigure:
    """Test the methods within a Figure object."""

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        """Generate random matrix.

        Returns
        -------
        pd.DataFrame
            Randomly generated matrix
        """
        n_samples, n_components = 1000, 50
        rng = random.default_rng()
        df = rng.standard_normal((n_samples, n_components))
        df = pd.DataFrame(df, columns=[f"IC{_+1}" for _ in range(n_components)])
        df.index.name = "Frame"
        return df

    @pytest.fixture
    def fig(self) -> figure.Figure:
        """Create a figure object.

        Returns
        -------
        Figure
            a figure object
        """
        return figure.Figure()

    def test_draw(self, data: pd.DataFrame, fig: figure.Figure) -> None:
        """Test draw method.

        GIVEN a 2D array with shape (n_samples, n_components)
        WHEN the draw method of a `Figure` object is called
        THEN a figure with subplots is created

        Parameters
        ----------
        data : pd.DataFrame
            Randomly generated matrix
        fig : Figure
            Figure object
        """
        fig.draw(data)

        assert isinstance(fig.figure, hv.Layout)
        assert fig.azimuth == 120

    def test_save(
        self,
        data: pd.DataFrame,
        fig: figure.Figure,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test save method.

        GIVEN a filename
        WHEN the save method method of a `Figure` object is called
        THEN an image is written to disk

        Parameters
        ----------
        data : pd.DataFrame
            Randomly generated matrix
        fig : Figure
            Figure object
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        filename: Path = tmp_path.joinpath("test.png")
        fig.draw(data)
        fig.save(filename.as_posix())
        assert filename.exists()
        assert filename.stat().st_size > 0
