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
from typing import Optional, Tuple

import numpy as np
from scipy import linalg

from .typing import Array

logger: logging.Logger = logging.getLogger(__name__)


def kabsch_rotate(P: Array, Q: Array, /) -> Array:
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N, D) matrix, where N is points and D is dimension.
    Q : array
        (N, D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N, D) matrix, where N is points and D is dimension,
        rotated
    """
    U: Array = kabsch(P, Q)

    # Rotate P
    P: Array = P @ U
    return P


def kabsch_fit(P: Array, Q: Array, /, W: Optional[Array] = None) -> Array:
    """Rotate and translate matrix P unto matrix Q using Kabsch algorithm.
    An optional vector of weights W may be provided.

    Parameters
    ----------
    P : array
        (N, D) matrix, where N is points and D is dimension.
    Q : array
        (N, D) matrix, where N is points and D is dimension.
    W : array or None
        (N,) vector, where N is points.

    Returns
    -------
    P : array
        (N, D) matrix, where N is points and D is dimension,
        rotated and translated.
    """
    if W is not None:
        P: Array = kabsch_weighted_fit(P, Q, W, rmsd=False)
    else:
        QC: Array = centroid(Q)
        Q: Array = Q - QC
        P: Array = P - centroid(P)
        P: Array = kabsch_rotate(P, Q) + QC
    return P


def kabsch(P: Array, Q: Array) -> Array:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : Array
        (N, D) array, where N is points and D is dimension.
    Q : Array
        (N, D) array, where N is points and D is dimension.

    Returns
    -------
    U : Array
        Rotation matrix (D, D)
    """
    # Computation of the covariance matrix
    C: Array = P.T @ Q

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = linalg.svd(C)
    d: int = (linalg.det(V) * linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U: Array = V @ W

    return U


def kabsch_weighted(P: Array, Q: Array, /, W: Optional[Array] = None) -> Array:
    """
    Using the Kabsch algorithm with two sets of paired point P and Q.
    Each vector set is represented as an NxD matrix, where D is the
    dimension of the space.
    An optional vector of weights W may be provided.
    Note that this algorithm does not require that P and Q have already
    been overlayed by a centroid translation.
    The function returns the rotation matrix U, translation vector V,
    and RMS deviation between Q and P', where P' is:
        P' = P * U + V
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : Array
        (N, D) matrix, where N is points and D is dimension.
    Q : Array
        (N, D) matrix, where N is points and D is dimension.
    W : Array or None
        (N, ) vector, where N is points.

    Returns
    -------
    U : matrix
           Rotation matrix (D,D)
    V : vector
           Translation vector (D)
    RMSD : float
           Root mean squared deviation between P and Q
    """
    # Computation of the weighted covariance matrix
    C: Array = np.zeros((3, 3))
    W: Array = np.ones_like(P) / len(P) if W is None else np.array([W, W, W]).T

    # NOTE UNUSED psq = 0.0
    # NOTE UNUSED qsq = 0.0
    iw: float = 3.0 / W.sum()
    n: int = len(P)
    for i, j, k in itertools.product(range(3), range(n), range(3)):
        C[i, k] += P[j, i] * Q[j, k] * W[j, i]
    CMP: Array = (P * W).sum(axis=0)
    CMQ: Array = (Q * W).sum(axis=0)
    PSQ: Array = (P * P * W).sum() - (CMP * CMP).sum() * iw
    QSQ: Array = (Q * Q * W).sum() - (CMQ * CMQ).sum() * iw
    C: Array = (C - np.outer(CMP, CMQ) * iw) * iw

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = linalg.svd(C)
    d: int = (linalg.det(V) * linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U, translation vector V, and calculate RMSD:
    U: Array = V @ W
    msd: float = (PSQ + QSQ) * iw - 2.0 * S.sum()
    msd = np.clip(msd, 0.0)
    rmsd = np.sqrt(msd)
    V: Array = np.zeros(3)
    for i in range(3):
        t: float = (U[i, :] * CMQ).sum()
        V[i] = CMP[i] - t
    V: Array = V * iw
    return U, V, rmsd


def kabsch_weighted_fit(
    P: Array, Q: Array, /, W: Optional[Array] = None, rmsd: bool = False
) -> Tuple[Array, float]:
    """
    Fit P to Q with optional weights W.
    Also returns the RMSD of the fit if rmsd=True.

    Parameters
    ----------
    P : Array
       (N, D) matrix, where N is points and D is dimension.
    Q : Array
       (N, D) matrix, where N is points and D is dimension.
    W : Array
       (N, ) vector, where N is points
    rmsd : bool
       If True, rmsd is returned as well as the fitted coordinates.

    Returns
    -------
    P' : Array
       (N, D) matrix, where N is points and D is dimension.
    RMSD : float
       if the function is called with rmsd=True
    """
    R, T, RMSD = kabsch_weighted(Q, P, W)
    PNEW: Array = (P @ R.T) + T
    if rmsd:
        return PNEW, RMSD
    else:
        return PNEW


def kabsch_weighted_rmsd(P: Array, Q: Array, /, W: Optional[Array] = None) -> float:
    """Calculate the RMSD between P and Q with optional weighhts W

    Parameters
    ----------
    P : Array
        (N, D) matrix, where N is points and D is dimension.
    Q : Array
        (N, D) matrix, where N is points and D is dimension.
    W : Array
        (N, ) vector, where N is points

    Returns
    -------
    RMSD : float
    """
    _, _, w_rmsd = kabsch_weighted(P, Q, W)
    return w_rmsd


def quaternion_transform(r: Array, /) -> Array:
    """Get optimal rotation

    note: translation will be zero when the centroids of each molecule are the same
    """
    Wt_r: Array = makeW(*r).T
    Q_r: Array = makeQ(*r)
    rot: Array = Wt_r.dot(Q_r)[:3, :3]
    return rot


def makeW(r1: float, r2: float, r3: float, r4: float = 0) -> Array:
    """matrix involved in quaternion rotation"""
    W: Array = np.asarray(
        [
            [r4, r3, -r2, r1],
            [-r3, r4, r1, r2],
            [r2, -r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return W


def makeQ(r1: float, r2: float, r3: float, r4: float = 0) -> Array:
    """matrix involved in quaternion rotation"""
    Q: Array = np.asarray(
        [
            [r4, -r3, r2, r1],
            [r3, r4, -r1, r2],
            [-r2, r1, r4, r3],
            [-r1, -r2, -r3, r4],
        ]
    )
    return Q


def quaternion_rotate(X: Array, Y: Array, /) -> Array:
    """Calculate the rotation

    Parameters
    ----------
    X : Array
        (N, D) matrix, where N is points and D is dimension.
    Y: Array
        (N, D) matrix, where N is points and D is dimension.

    Returns
    -------
    rot : Array
        Rotation matrix (D, D)
    """
    N: int = X.shape[0]
    W: Array = np.asarray([makeW(*Y[k]) for k in range(N)])
    Q: Array = np.asarray([makeQ(*X[k]) for k in range(N)])
    Qt_dot_W: Array = np.asarray([np.dot(Q[k].T, W[k]) for k in range(N)])
    # NOTE UNUSED W_minus_Q = np.asarray([W[k] - Q[k] for k in range(N)])
    A: Array = np.sum(Qt_dot_W, axis=0)
    eval, evec = linalg.eigh(A)
    r: Array = evec[:, eval[0].argmax()]
    rot: Array = quaternion_transform(r)
    return rot


def centroid(X: Array) -> float:
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.
    https://en.wikipedia.org/wiki/Centroid

    ..math::
        C = \langle X \rangle

    Parameters
    ----------
    X : Array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    C : float
        centroid
    """
    return X.mean(axis=0)
