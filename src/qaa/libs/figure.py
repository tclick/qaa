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
from typing import Optional

import holoviews as hv
import pandas as pd
from holoviews import opts

from .. import PathLike

hv.extension("matplotlib")


class Figure:
    """Create a plot of 2D and 3D plots."""

    def __init__(
        self,
        *,
        azim: int = 120,
    ) -> None:
        """Visualize data via a graphical image.

        Parameters
        ----------
        azim : int
            Azimuth rotation for 3D plot
        """
        self._figure: Optional[hv.Layout] = None
        self._azimuth: int = azim

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
        try:
            columns = data.drop("Cluster", axis=1).columns[:3].to_list()
        except KeyError:
            columns = data.columns[:3].to_list()

        if "Cluster" not in columns:
            scatter = [
                hv.Scatter(data, kdims=i, vdims=j)
                for i, j in itertools.combinations(columns, 2)
            ]
            self._figure = hv.Layout(scatter)
            self._figure += hv.Scatter3D(data, kdims=columns, vdims=[])
            self._figure.opts(
                opts.Scatter(marker=".", s=10),
                opts.Scatter3D(azimuth=self._azimuth, marker=".", s=10),
            )
        else:
            if centers is None:
                scatter = [
                    hv.Scatter(data, kdims=i, vdims=[j, "Cluster"])
                    for i, j in itertools.combinations(columns, 2)
                ]
            else:
                scatter = [
                    (
                        hv.Scatter(data, kdims=i, vdims=[j, "Cluster"])
                        * hv.Scatter(centers, kdiims=i, vdims=[j, "Cluster"])
                    )
                    for i, j in itertools.combinations(columns, 2)
                ]

            self._figure = hv.Layout(scatter)
            self._figure.opts(
                opts.Scatter(
                    show_legend=False,
                    color_index="Cluster",
                    color=hv.Palette("Dark2"),
                    marker=".",
                    s=10,
                ),
                opts.Scatter(
                    show_legend=True,
                    color_index="Cluster",
                    color=hv.Palette("tab20"),
                    marker=".",
                    s=20,
                ),
                opts.Scatter3D(
                    show_legend=False,
                    color_index="Cluster",
                    color=hv.Palette("Dark2"),
                    azimuth=self._azimuth,
                    marker=".",
                    s=10,
                ),
            )

        #         if centers is not None:
        #             clusters = hv.Scatter(centers, kdiims=i, vdims=[j, "Cluster"])
        #             clusters.opts(
        #                 show_legend=False,
        #                 color_index="Cluster",
        #                 color=hv.Palette("tab20"),
        #             )
        #             fig *= clusters
        #         scatter.append(fig)
        #
        # scatter3d = hv.Scatter3D(table, kdims=columns, vdims="Cluster")
        # scatter3d.opts(opts.Scatter3D(azimuth=self._azimuth))

        # if centers is not None:
        #     scatter3d *= hv.Scatter3D(centers, kdims=columns, vdims="Cluster").opts(
        #         show_legend=True, color_index="Cluster", color=hv.Palette("Dark2")
        #     )

        # self._figure = hv.Layout(scatter) + scatter3d
        self._figure.cols(2)

    def save(
        self, filename: PathLike, /, *, dpi: int = 600, title: Optional[str] = None
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
            backend="matplotlib",
            title=title,
        )
