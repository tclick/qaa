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
from typing import Optional

import holoviews as hv
import pandas as pd
from colorcet import glasbey_cool
from colorcet import glasbey_light

from .. import PathLike

hv.extension("matplotlib")

_empty_dataframe = pd.DataFrame()


class Figure:
    """Create a plot of 2D and 3D plots."""

    def __init__(
        self,
        *,
        azim: int = 120,
        elevation: int = 30,
    ) -> None:
        """Visualize data via a graphical image.

        Parameters
        ----------
        azim : int
            Azimuth rotation for 3D plot
        elevation : int
            Elevation of 3D plot
        """
        self._figure: Optional[hv.Layout] = None
        self._azimuth: int = azim
        self._elevation: int = elevation

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

    @property
    def elevation(self) -> int:
        """Set elevation of the 3D scatter plot.

        Returns
        -------
        int
            Elevation of the 3D plot
        """
        return self._elevation

    @elevation.setter
    def elevation(self, elevation: int) -> None:
        """Set elevation of the 3D scatter plot.

        Parameters
        ----------
        elevation : int
            Elevation
        """
        self._elevation = elevation

    def draw(
        self,
        data: pd.DataFrame,
        *,
        centers: pd.DataFrame = _empty_dataframe,
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
        table = hv.Table(data)
        kdims = table.data.columns[2:5].to_list()
        vdims = table.data.columns[:2].to_list()
        centroid = hv.Table(centers) if not centers.empty else hv.Table([])

        points = [
            hv.Points(table, [i, j], vdims=vdims).opts(
                show_legend=False,
                marker=".",
                s=5,
                cmap=glasbey_cool,
                color_index="Cluster",
            )
            for i, j in itertools.combinations(kdims, 2)
        ]
        if centroid:
            scatter = [
                hv.Points(centroid, kdims=[i, j], vdims="Cluster").opts(
                    show_legend=False,
                    marker=".",
                    s=15,
                    cmap=glasbey_light,
                    color_index="Cluster",
                )
                for i, j in itertools.combinations(kdims, 2)
            ]
            points = [i * j for i, j in zip(points, scatter)]
        self._figure = hv.Layout(points)

        scatter3d = hv.Scatter3D(data, kdims, vdims=vdims).opts(
            show_legend=False,
            marker=".",
            s=5,
            cmap=glasbey_cool,
            color_index="Cluster",
        )
        if centroid:
            scatter3d *= hv.Scatter3D(centers, kdims=kdims, vdims="Cluster").opts(
                show_legend=False,
                marker=".",
                s=25,
                cmap=glasbey_light,
                color_index="Cluster",
            )
        self._figure += scatter3d
        self._figure.cols(2)

    def save(
        self, filename: PathLike, *, dpi: int = 600, title: Optional[str] = None
    ) -> None:
        """Save the image to disk.

        Parameters
        ----------
        filename : PathLike
            Image file
        dpi : int
            Image resolution
        title : str
            Figure title
        """
        hv.output(dpi=dpi)
        hv.save(
            self._figure,
            filename=filename,
            fmt=Path(filename).suffix[1:],
            backend="matplotlib",
            title=title,
        )
