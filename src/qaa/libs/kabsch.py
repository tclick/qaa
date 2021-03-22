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
import itertools
import logging
from typing import Optional, Tuple, NoReturn

import dask.array as da
import numpy as np
from scipy import linalg

from .typing import ArrayType

logger: logging.Logger = logging.getLogger(__name__)

def kabsch_rotate(arr1: ArrayType, arr2: ArrayType, /) -> ArrayType:
    """Rotate matrix :math:`arr1` unto matrix :math:`arr2` using Kabsch algorithm.

    Parameters
    ----------
    arr1, arr2 : array_like
        (N, D) matrix, where N is points and D is dimension.

    Returns
    -------
    Array
        arr1 rotated onto arr2
    """
    rot_mat: ArrayType = kabsch(arr1, arr2)

    # Rotate arr1
    return arr1 @ rot_mat


def kabsch_fit(
    arr1: ArrayType, arr2: ArrayType, /, weight: Optional[ArrayType] = None
) -> ArrayType:
    """Rotate and translate matrix `P` unto matrix `Q` using Kabsch algorithm.

    An optional vector of weights W may be provided.

    Parameters
    ----------
    arr1, arr2 : Array
        Arrays with shape (n_points, n_dims)
    weight : Array or None
        Vector with shape (n_points,)

    Returns
    -------
    Array
        Rotated and translated array with shape (n_points, n_dims)
    """
    if weight is not None:
        arr1: ArrayType = kabsch_weighted_fit(arr1, arr2, weight, rmsd=False)
    else:
        qc: ArrayType = centroid(arr2)
        arr2: ArrayType = arr2 - qc
        arr1: ArrayType = arr1 - centroid(arr1)
        arr1: ArrayType = kabsch_rotate(arr1, arr2) + qc
    return arr1


def kabsch(arr1: ArrayType, arr2: ArrayType) -> ArrayType:
    """Align `arr1` to `arr2` using the Kabsch algorithm.

    For more info see https://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    arr1, arr2 : array_like
        Arrays with shape (n_points, n_dims)

    Returns
    -------
    rot_matrix : array_like
        Rotation matrix of shape (n_dims, n_dims)

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
    covar: ArrayType = arr1.T @ arr2

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    u, s, w = linalg.svd(covar)
    det: int = (linalg.det(u) * linalg.det(w)) < 0.0

    if det:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    # Create Rotation matrix U
    return u @ w


def kabsch_weighted(
    arr1: ArrayType, arr2: ArrayType, /, *, weight: Optional[ArrayType] = None
) -> ArrayType:
    """Align `arr1` to `arr2` using a weighted Kabsch method.

    Parameters
    ----------
    arr1, arr2 : array_like
        Arrays with shape (n_points, n_dims)
    weight : ArrayType or None
        Weights vector of shape (n_points)

    Returns
    -------
    U : matrix
           Rotation matrix (D, D)
    V : vector
           Translation vector (:math:`D`)
    rmsd : float
           Root mean squared deviation between `P` and `Q`

    Notes
    -----
    Each vector set is represented with the shape (n_points, n_dims). An optional vector
    of weights W may be provided. Note that this algorithm does not require that `P` and
    `Q` have already been overlaid by a centroid translation. The function returns the
    rotation matrix U, translation vector `V`, and RMS deviation between `Q` and `P'`,
    where P' is:

    .. math:: P' = P * U + V

    For more info see `http://en.wikipedia.org/wiki/Kabsch_algorithm`_
    """
    # Determine whether to use dask.array or scipy for linear algebra
    weights: ArrayType
    if isinstance(arr1, da.array) or isinstance(arr2, da.array):
        from dask.array import linalg

        weight: ArrayType = da.ones_like(arr1) / len(arr1) if weight is None else da.array([weight, weight, weight]).T
    else:
        from scipy import linalg

        weight: ArrayType = np.ones_like(arr1) / len(arr1) if weight is None else np.array([weight, weight, weight]).T

    # Computation of the weighted covariance matrix
    covar: ArrayType = np.zeros((3, 3))

    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw: float = 3.0 / weight.sum()
    n: int = len(arr1)
    for i, j, k in itertools.product(range(3), range(n), range(3)):
        covar[i, k] += arr1[j, i] * arr2[j, k] * weight[j, i]

    cmp: ArrayType = (arr1 * weight).sum(axis=0)
    cmq: ArrayType = (arr2 * weight).sum(axis=0)
    psq: ArrayType = (arr1 * arr1 * weight).sum() - (cmp * cmp).sum() * iw
    qsq: ArrayType = (arr2 * arr2 * weight).sum() - (cmq * cmq).sum() * iw
    covar: ArrayType = (covar - np.outer(cmp, cmq) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    u, s, vt = linalg.svd(covar)
    d: int = (linalg.det(u) * linalg.det(weight)) < 0.0

    if d:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    v: ArrayType = u @ vt
    msd: float = (psq + qsq) * iw - 2.0 * s.sum()
    msd = np.clip(msd, 0.0)
    rmsd = np.sqrt(msd)
    u: ArrayType = np.zeros(3)
    for i in range(3):
        t: float = (v[i, :] * cmq).sum()
        u[i] = cmp[i] - t
    u: ArrayType = u * iw
    return v, u, rmsd


def kabsch_weighted_fit(
    arr1: ArrayType, arr2: ArrayType, /, weight: Optional[ArrayType] = None, rmsd: bool = False
) -> Tuple[ArrayType, float]:
    """Fit `arr1` to `arr2` with optional weights `weight`.

    Also returns the RMSD of the fit if rmsd=True.

    Parameters
    ----------
    arr1, arr2 : array_like
        Arrays with shape (n_points, n_dims)
    weight : ArrayType or None
        Weights vector of shape (n_points)
    rmsd : bool
       If True, rmsd is returned

    Returns
    -------
    aligned : ArrayType
       Translated and rotated array of shape (n_points, n_dims)
    rmsd : float
       Root mean squared deviation between `P` and `Q`
    """
    rotation, translation, rmsd = kabsch_weighted(arr2, arr1, weight=weight)
    aligned: ArrayType = (arr1 @ rotation.T) + translation
    if rmsd:
        return aligned, rmsd
    else:
        return aligned


def kabsch_weighted_rmsd(
    P: ArrayType, Q: ArrayType, /, W: Optional[ArrayType] = None
) -> float:
    """Calculate the RMSD between P and Q with optional weighhts W

    Parameters
    ----------
    P : ArrayType
        (N, D) matrix, where N is points and D is dimension.
    Q : ArrayType
        (N, D) matrix, where N is points and D is dimension.
    W : ArrayType
        (N, ) vector, where N is points

    Returns
    -------
    RMSD : float
    """
    _, _, w_rmsd = kabsch_weighted(P, Q, W)
    return w_rmsd


def quaternion_transform(r: ArrayType, /) -> ArrayType:
    """Get optimal rotation

    note: translation will be zero when the centroids of each molecule are the same
    """
    Wt_r: ArrayType = makeW(*r).T
    Q_r: ArrayType = makeQ(*r)
    rot: ArrayType = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1: float, r2: float, r3: float, r4: float = 0) -> ArrayType:
    """matrix involved in quaternion rotation"""
    W: ArrayType = np.asarray(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return W


def makeQ(r1: float, r2: float, r3: float, r4: float = 0) -> ArrayType:
    """matrix involved in quaternion rotation"""
    Q: ArrayType = np.asarray(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return Q


def quaternion_rotate(X: ArrayType, Y: ArrayType, /) -> ArrayType:
    """Calculate the rotation

    Parameters
    ----------
    X : ArrayType
        (N, D) matrix, where N is points and D is dimension.
    Y: ArrayType
        (N, D) matrix, where N is points and D is dimension.

    Returns
    -------
    rot : ArrayType
        Rotation matrix (D, D)
    """
    N: int = X.shape[0]
    W: ArrayType = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q: ArrayType = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W: ArrayType = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    # NOTE UNUSED W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A: ArrayType = np.sum(Qt_dot_W, axis=0)
    eval, evec = linalg.eigh(A)
    r: ArrayType = evec[:, eval[0].argmax()]
    rot: ArrayType = quaternion_transform(r)
    return rot


def centroid(X: ArrayType) -> float:
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.
    https://en.wikipedia.org/wiki/Centroid

    ..math::
        C = :math:`\langle` X :math:`\rangle`

    Parameters
    ----------
    X : ArrayType
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    C : float
        centroid
    """
    return X.mean(axis=0)
