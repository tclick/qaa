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
"""Align a trajectory using the Kabsch method.

Notes
-----
Kabsch functions originally from
https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Optional
from typing import Tuple

from nptyping import Float
from nptyping import NDArray
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from .utils import rmse

logger: logging.Logger = logging.getLogger(__name__)


class Kabsch(TransformerMixin, BaseEstimator):
    """Kabsch alignment method."""

    def __init__(self, verbose: bool = True) -> None:
        """Align one array to another using the Kabsch method.

        Parameters
        ----------
        verbose : bool
            Print data to the log
        """
        self.rotation_: NDArray[(Any, ...), Float]
        self.translation_: NDArray[(Any, ...), Float]
        self.error: float
        self.verbose = verbose

    def fit(
        self, mobile: NDArray[(Any, ...), Float], reference: NDArray[(Any, ...), Float]
    ) -> Kabsch:
        """Center and fit `mobile` onto `reference` using the Kabsch method.

        Parameters
        ----------
        mobile, reference : array_like
            Arrays with shape (n_points, 3)

        Returns
        -------
        self
        """
        mobile = check_array(mobile, copy=self.verbose)
        reference = check_array(reference, copy=self.verbose)
        new_mobile, _ = self._center(mobile)
        new_ref, ref_mean = self._center(reference)

        self.rotation_ = self._fit(new_mobile, new_ref)
        self.translation_ = ref_mean.copy()

        return self

    def transform(
        self,
        mobile: NDArray[(Any, ...), Float],
        reference: Optional[NDArray[(Any, ...), Float]] = None,
    ) -> NDArray[(Any, ...), Float]:
        """Align `mobile` to `reference`.

        Parameters
        ----------
        mobile, reference : array_like
            Arrays with shape (n_points, 3)

        Returns
        -------
        array_like
            Array with shape (n_points, 3)
        """
        check_is_fitted(self)

        mobile = StandardScaler(with_std=False).fit_transform(mobile)
        new_mobile: NDArray[(Any, ...), Float] = (
            mobile @ self.rotation_ + self.translation_
        )
        self.error = rmse(mobile, new_mobile)
        return new_mobile

    def _center(
        self, positions: NDArray[(Any, ...), Float]
    ) -> Tuple[NDArray[(Any, ...), Float], NDArray[(Any, ...), Float]]:
        scale = StandardScaler(with_std=False)
        new_positions: NDArray[(Any, ...), Float] = scale.fit_transform(positions)
        return new_positions, scale.mean_

    def _fit(
        self, mobile: NDArray[(Any, ...), Float], reference: NDArray[(Any, ...), Float]
    ) -> NDArray[(Any, ...), Float]:
        """Align `mobile` to `reference` using the Kabsch algorithm.

        For more info see https://en.wikipedia.org/wiki/Kabsch_algorithm

        Parameters
        ----------
        mobile, reference : NDArray[(Any, ...), Float]
            Arrays with shape (n_points, 3)

        Returns
        -------
        ArrayType
            Rotation matrix with shape (3, 3)

        Notes
        -----
        Using the Kabsch algorithm with two sets of paired point `arr1` and `arr2`, centered
        around the centroid. Each vector set is represented with the shape  (n_points,
        n_dims) where n_dims is the dimension of the space.  The algorithm works in three
        steps:

        * a centroid translation of `arr1` and `arr2` (completed before this function call)
        * the computation of a covariance matrix `covar`
        * computation of the optimal rotation matrix `rot_matrix`
        """
        # Computation of the covariance matrix
        covar: NDArray[(Any, ...), Float] = mobile.T @ reference

        # Computation of the optimal rotation matrix
        # This can be done using singular value decomposition (SVD)
        # Getting the sign of the det(V)*(W) to decide
        # whether we need to correct our rotation matrix to ensure a
        # right-handed coordinate system.
        # And finally calculating the optimal rotation matrix U
        # see http://en.wikipedia.org/wiki/Kabsch_algorithm
        u, s, vh = linalg.svd(covar)
        if (linalg.det(u) * linalg.det(vh)) < 0.0:
            s[-1] = -s[-1]
            u[:, -1] = -u[:, -1]

        return u @ vh
