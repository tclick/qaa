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
"""Draw and save figures for QAA."""
import itertools
from typing import NoReturn, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from .typing import ArrayType, PathLike


class Figure:
    def __init__(self, method: str = "ica"):
        """Visualize data via a graphical image

        Parameters
        ----------
        method : {'ica', 'pca'}
            Type of data
        """
        self.method = method
        self._figure: Optional[plt.Figure] = None
        self._axes: Optional[plt.Axes] = None

    @property
    def figure(self) -> plt.Figure:
        """Return the underlying figure object."""
        return self._figure

    @figure.setter
    def figure(self, fig: plt.Figure) -> NoReturn:
        """Set the underlying figure."""
        self._figure = fig

    @property
    def axes(self) -> plt.Axes:
        """Return the underlying axes object."""
        return self._axes

    @axes.setter
    def axes(self, ax: plt.Axes) -> NoReturn:
        """Set the underlying axes."""
        self._axes = ax

    def draw(self, data: ArrayType) -> NoReturn:
        """Draw the first three components in subplots.

        Parameters
        ----------
        data : array_like
            Matrix with shape (n_samples, n_components)

        Notes
        -----
        Four subplots are created.
        * component 1 vs. component 2
        * component 1 vs. component 3
        * component 2 vs. component 3
        * 3D plot
        """
        self._figure: plt.Figure = plt.figure(figsize=plt.figaspect(1.0))
        data_type: str = self.method.upper()
        label: str = data_type[:2]

        for i, (x, y) in enumerate(itertools.combinations(range(3), 2), 1):
            self._axes = self._figure.add_subplot(2, 2, i)
            sns.scatterplot(x=data[:, x], y=data[:, y], ax=self._axes, marker=".")
            self._axes.set_xlabel(f"${label}_{x + 1:d}$")
            self._axes.set_ylabel(f"${label}_{y + 1:d}$")

        # Plot first 3 PCs
        self._axes = self._figure.add_subplot(
            2, 2, i + 1, projection="3d", proj_type="ortho"
        )
        self._axes.scatter3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            marker=".",
            alpha=0.5,
        )
        self._axes.view_init(azim=120)
        self._axes.set_xlabel(f"${label}_1$")
        self._axes.set_ylabel(f"${label}_2$")
        self._axes.set_zlabel(f"${label}_3$")
        self._figure.suptitle(f"{data_type}")
