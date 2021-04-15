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
from typing import Any
from typing import Optional

import holoviews as hv
import pandas as pd
from holoviews import opts
from nptyping import Float
from nptyping import NDArray

from .. import PathLike

hv.extension("matplotlib")


class Figure:
    """Create a plot of 2D and 3D plots."""

    def __init__(
        self,
        *,
        n_points: Optional[int] = None,
        method: str = "ica",
        labels: Optional[NDArray[(Any, ...), Float]] = None,
        azim: Optional[int] = None,
    ) -> None:
        """Visualize data via a graphical image.

        Parameters
        ----------
        n_points : int
            Number of points to include for 3D plots
        method : str
            Type of data
        labels : NDArray[(Any, ...), Float]
            Vector of cluster labels
        azim : int
            Azimuth rotation for 3D plot
        """
        self.n_points = n_points
        self.method = method
        self.labels = labels
        self._figure: hv.Layout
        self._azimuth: int = azim if azim is not None else 120

    @property
    def figure(self) -> hv.Layout:
        """Return a holoviews Layout.

        Returns
        -------
        hv.Layout
            A multicolumn 2D and 3D scatter plots
        """
        return self._figure

    @property
    def azimuth(self) -> int:
        """Return the azimuth of the 3D scatter plot.

        Returns
        -------
        int
            Azimuth
        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth: int) -> None:
        """Set azimuth of the 3D scatter plot.

        Parameters
        ----------
        azimuth : int
            Azimuth
        """
        self._azimuth = azimuth

    def draw(
        self,
        data: pd.DataFrame,
        /,
        *,
        centers: Optional[pd.DataFrame] = None,
    ) -> None:
        """Draw the first three components in subplots.

        Parameters
        ----------
        data : pd.DataFrame
            Matrix with shape (n_samples, n_components)
        centers : pd.DataFrame, optional
            Vector of cluster centers (n_components, )

        Notes
        -----
        Four subplots are created.
        * component 1 vs. component 2
        * component 1 vs. component 3
        * component 2 vs. component 3
        * 3D plot
        """
        header = self.method[:2].upper()
        scatter = []
        for i, j in itertools.combinations(range(3), 2):
            fig = hv.Scatter(data, f"{header}{i+1}", f"{header}{j+1}")
            if centers is not None:
                fig *= hv.Scatter(centers, f"{header}{i+1}", f"{header}{j+1}")
            scatter.append(fig)

        scatter3d = hv.Scatter3D(data, kdims=[f"{header}1", f"{header}2", f"{header}3"])
        scatter3d.opts(opts.Scatter3D(azimuth=self._azimuth))
        if centers is not None:
            scatter3d *= hv.Scatter3D(
                centers, kdims=[f"{header}1", f"{header}2", f"{header}3"]
            )

        self._figure = hv.Layout(scatter) + scatter3d
        self._figure.cols(2)

    def save(self, filename: PathLike, /, *, dpi: int = 600) -> None:
        """Save the image to disk.

        Parameters
        ----------
        filename : PathLike
            Image file
        dpi : int
            Image resolution
        """
        hv.output(dpi=dpi)
        hv.save(
            self._figure,
            filename=filename,
            backend="matplotlib",
            title=f"First three {self.method[:2].upper()}s",
        )
