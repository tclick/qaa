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
from pathlib import Path
from typing import NoReturn
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .typing import ArrayType
from .typing import PathLike


class Figure:
    """Create a plot of 2D and 3D plots."""

    def __init__(
        self,
        *,
        n_points: int = 10,
        method: str = "ica",
        labels: Optional[ArrayType] = None,
        azim: float = 120.0,
    ):
        """Visualize data via a graphical image.

        Parameters
        ----------
        n_points : int
            Number of points to include for 3D plots
        method : str
            Type of data
        labels : ArrayType
            Vector of cluster labels
        azim : float
            Azimuth rotation for 3D plot
        """
        self.n_points = n_points
        self.method = method
        self.labels = labels
        self._cmap = (
            sns.husl_palette(n_colors=np.unique(labels).size, as_cmap=True)
            if labels is not None
            else None
        )
        self._figure: Optional[plt.Figure] = None
        self._axes: Optional[plt.Axes] = None
        self._azim: float = azim

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

    def draw(
        self, data: ArrayType, /, *, centers: Optional[ArrayType] = None
    ) -> NoReturn:
        """Draw the first three components in subplots.

        Parameters
        ----------
        data : ArrayType
            Matrix with shape (n_samples, n_components)
        centers : ArrayType
            Vector of cluster centers (n_components, )

        Notes
        -----
        Four subplots are created.
        * component 1 vs. component 2
        * component 1 vs. component 3
        * component 2 vs. component 3
        * 3D plot
        """
        sns.set_theme(context="paper", style="ticks", palette="husl")
        self._figure: plt.Figure = plt.figure(figsize=plt.figaspect(1.0))
        data_type: str = self.method.upper()
        label: str = data_type[:2]

        for i, (x, y) in enumerate(itertools.combinations(range(3), 2), 1):
            self._axes = self._figure.add_subplot(2, 2, i)
            sns.scatterplot(
                x=data[:, x],
                y=data[:, y],
                ax=self._axes,
                marker=".",
                hue=self.labels,
                edgecolor="none",
                legend=False,
            )
            if centers is not None:
                sns.scatterplot(
                    x=centers[:, x],
                    y=centers[:, y],
                    ax=self._axes,
                    marker="o",
                    palette=self._cmap,
                    hue=np.unique(self.labels),
                    edgecolor="none",
                )
            self._axes.set_xlabel(f"${label}_{x + 1:d}$")
            self._axes.set_ylabel(f"${label}_{y + 1:d}$")

        # Plot first 3 PCs
        self._axes = self._figure.add_subplot(
            2, 2, i + 1, projection="3d", proj_type="ortho"
        )

        self._axes.scatter3D(
            data[:: self.n_points, 0],
            data[:: self.n_points, 1],
            data[:: self.n_points, 2],
            marker=".",
            cmap=self._cmap,
            c=self.labels[:: self.n_points],
            alpha=0.5,
        )
        if centers is not None:
            self._axes.scatter3D(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                marker="o",
                cmap=sns.husl_palette(n_colors=len(centers), as_cmap=True),
                c=np.arange(len(centers)),
            )

        self._axes.view_init(azim=self._azim)
        self._axes.set_xlabel(f"${label}_1$")
        self._axes.set_ylabel(f"${label}_2$")
        self._axes.set_zlabel(f"${label}_3$")
        self._figure.suptitle(f"{data_type}")
        self._figure.tight_layout()

    def save(self, filename: PathLike, /, *, dpi: int = 600) -> NoReturn:
        """Save the image to disk.

        Parameters
        ----------
        filename : PathLike
            Image file
        dpi : int
            Image resolution
        """
        with Path(filename).open(mode="wb") as w:
            self._figure.savefig(w, dpi=dpi)
