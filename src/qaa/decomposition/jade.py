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
"""Module to find signals using the Jade ICA method.

This module contains the function, _jade, which does blind source separation of real
signals. The original Python code can be found at
https://github.com/gvacaliuc/jade_c/blob/master/jade.py
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Optional

import numpy as np
from nptyping import Float
from nptyping import NDArray
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


def _jade(
    arr: NDArray[(Any, ...), Float], n_components: Optional[int] = None
) -> NDArray[(Any, ...), Float]:
    """Blind separation of real signals with JADE.

    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm developed
    by Jean-Francois Cardoso. See http://www.tsi.enst.fr/~cardoso/guidesepsou.html, and
    papers cited at the end of the source file.

    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by Gabriel
    Beckers, http://gbeckers.nl .

    Parameters
    ----------
    arr : NDArray
        a data matrix (n_features, n_samples)
    n_components : int, default=None
        output matrix B has size mxn so that only m sources are extracted.  This is done
        by restricting the operation of jadeR to the m first principal components.
        Defaults to None, in which case :math:`m = n`.

    Returns
    -------
    NDArray
        An m*n matrix B (NumPy matrix type), such that Y=B*X are separated sources
        extracted from the n*T data matrix X. If m is omitted, B is a square n*n matrix
        (as many sources as sensors). The rows of B are ordered such that the columns of
        :math:`pinv(B)` are in order of decreasing norm; this has the effect that the
        `most energetically significant` components appear first in the rows of
        :math:`Y = B * X`.

    Raises
    ------
    IndexError
        if :math:`m > n_features`

    Notes
    -----
    Quick notes (more at the end of this file):

    - This code is for REAL-valued signals.  A MATLAB implementation of JADE for both
      real and complex signals is also available from
      http://sig.enst.fr/~cardoso/stuff.html

    - This algorithm differs from the first released implementations of JADE in that it
      has been optimized to deal more efficiently
        1) with real signals (as opposed to complex)
        2) with the case when the ICA model does not necessarily hold.

    - There is a practical limit to the number of independent components that can be
      extracted with this implementation.  Note that the first step of JADE amounts to a
      PCA with dimensionality reduction from `n` to `m` (which defaults to n).  In
      practice m cannot be *very large* (more than 40, 50, 60... depending on available
      memory)

    - See more notes, references and revision history at the end of this file and more
      stuff at http://sig.enst.fr/~cardoso/stuff.html

    - For more info on NumPy translation, see the end of this file.

    - This code is supposed to do a good job!  Please report any problem relating to
      the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code : Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation : Gabriel Beckers <gabriel@gbeckers.nl>
    """
    # GB: we do some checking of the input arguments and copy data to new variables to
    # avoid messing with the original input. We also require double precision (float64)
    # and a numpy matrix type for X.
    origtype = arr.dtype  # remember to return matrix B of the same type
    arr = check_array(arr, dtype=float)

    # GB: n is number of input signals, T is number of samples
    n_samples, n_features = arr.shape

    if n_components is None:
        n_components = n_features  # Number of sources defaults to # of sensors
    elif n_components > n_features:
        raise IndexError(f"More sources ({n_components}) than sensors ({n_features})")

    logger.info("jade -> Looking for %d sources", n_components)
    logger.info("jade -> Removing the mean value")
    arr -= arr.mean(axis=0)

    # whitening & projection onto signal subspace
    # ===========================================
    logger.info("jade -> Whitening the data")

    # --- PCA  ----------------------------------------------------------
    u, s, vh = linalg.svd(arr, full_matrices=False)
    u, s, vh = u[:, :n_components], s[:n_components], vh[:n_components]
    B = linalg.inv(np.diag(s)) @ vh * np.sqrt(n_samples)
    arr = u * np.sqrt(n_samples)

    del u, s, vh
    # NOTE: At this stage, X is a PCA analysis in m components of the real data, except
    # that all its entries now have unit variance.  Any further rotation of X will
    # preserve the property that X is a vector of uncorrelated components.  It remains
    # to find the rotation matrix such that the entries of X are not only uncorrelated
    # but also `as independent as possible".  This independence is measured by
    # correlations of order higher than 2.  We have defined such a measure of
    # independence which
    #   1) is a reasonable approximation of the mutual information
    #   2) can be optimized by a `fast algorithm"
    # This measure of independence also corresponds to the `diagonality" of a set of
    # cumulant matrices.  The code below finds the `missing rotation " as the matrix
    # which best diagonalizes a particular set of cumulant matrices.

    # Estimation of the cumulant matrices.
    # ====================================
    logger.info("jade -> Estimating cumulant matrices")

    # Dim. of the space of real symm matrices
    dimsymm = n_components * (n_components + 1) // 2
    nbcm = dimsymm  # number of cumulant matrices
    # Storage for cumulant matrices
    CM = np.zeros((n_components, n_components * nbcm))
    R = np.eye(n_components)

    # I am using a symmetry trick to save storage.  I should write a short note one of
    # these days explaining what is going on here.
    # will index the columns of CM where to store the cum. mats.
    Range = np.arange(n_components)

    for im in range(n_components):
        Xim = np.vstack(arr[:, im])
        Xijm = Xim ** 2
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        Rim = np.vstack(R[:, im])
        Qij = (Xijm * arr).T @ arr / n_samples - R - 2 * (Rim @ Rim.T)
        CM[:, Range] = Qij
        Range = Range + n_components
        for jm in range(im):
            Xijm = Xim * np.vstack(arr[:, jm])
            Rjm = np.vstack(R[:, jm])
            CM[:, Range] = (
                np.sqrt(2) * (Xijm * arr).T @ arr / n_samples
                - Rim @ Rjm.T
                - Rjm @ Rim.T
            )
            Range = Range + n_components

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.

    V = np.eye(n_components)

    On = 0.0
    Range = np.arange(n_components)
    for _ in range(nbcm):
        diag = np.diag(CM[:, Range])
        On += (diag * diag).sum(axis=0)
        Range += n_components
    Off = ((CM * CM).sum(axis=0)).sum(axis=0) - On

    # % A statistically scaled threshold on `small" angles
    seuil = 1e-6 / np.sqrt(n_samples)
    encore = True
    sweep = 0  # % sweep number
    updates = 0  # % Total number of rotations

    # Joint diagonalization proper
    logger.info("jade -> Contrast optimization by joint diagonalization")

    counters = tuple(
        (p, q) for p in range(n_components - 1) for q in range(p + 1, n_components)
    )
    while encore:
        encore = False
        logger.info("jade -> Sweep #%3d", sweep)
        sweep += 1
        upds = 0  # % Number of rotations in a given seep

        for p, q in counters:
            Ip = np.arange(p, n_components * nbcm, n_components)
            Iq = np.arange(q, n_components * nbcm, n_components)

            # computation of Givens angle
            g = np.vstack([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
            gg = g @ g.T
            ton = gg[0, 0] - gg[1, 1]
            toff = gg[0, 1] + gg[1, 0]
            theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton * ton + toff * toff))
            Gain = (np.sqrt(ton * ton + toff * toff) - ton) / 4.0

            # Givens update
            if abs(theta) > seuil:
                encore = True
                upds += 1
                c = np.cos(theta)
                s = np.sin(theta)
                G = np.array([[c, -s], [s, c]])
                pair = np.array([p, q])
                V[:, pair] = V[:, pair] @ G
                CM[pair, :] = G.T @ CM[pair, :]
                CM[:, np.concatenate([Ip, Iq])] = np.append(
                    c * CM[:, Ip] + s * CM[:, Iq],
                    -s * CM[:, Ip] + c * CM[:, Iq],
                    axis=1,
                )
                On += Gain
                Off -= Gain

        logger.info("completed in %d rotations", upds)
        updates = updates + upds
    logger.info("jade -> Total of %d Givens rotations", updates)

    # A separating matrix
    # ===================
    B = V.T @ B

    # Permute the rows of the separating matrix B to get the most energetic components
    # first. Here the **signals** are normalized to unit variance.  Therefore, the sort
    # is according to the norm of the columns of A = pinv(B)

    logger.info("jade -> Sorting the components")

    A = linalg.pinv(B)
    keys = np.argsort((A * A).sum(axis=0))[::-1]
    B = B[keys, :]

    logger.info("jade -> Fixing the signs")
    b = B[:, 0]
    signs = np.sign(np.sign(b) + 0.1)  # just a trick to deal with sign=0
    B = np.diag(signs) @ B

    return B.astype(origtype)

    # Revision history of MATLAB code:
    #
    # - V1.8, May 2005
    #  - Added some commented code to explain the cumulant computation tricks.
    #  - Added reference to the Neural Comp. paper.
    #
    # -  V1.7, Nov. 16, 2002
    #   - Reverted the mean removal code to an earlier version (not using
    #     repmat) to keep the code octave-compatible.  Now less efficient,
    #     but does not make any significant difference wrt the total
    #     computing cost.
    #   - Remove some cruft (some debugging figures were created.  What
    #     was this stuff doing there???)
    #
    #
    # -  V1.6, Feb. 24, 1997
    #   - Mean removal is better implemented.
    #   - Transposing X before computing the cumulants: small speed-up
    #   - Still more comments to emphasize the relationship to PCA
    #
    # -  V1.5, Dec. 24 1997
    #   - The sign of each row of B is determined by letting the first element be
    #     positive.
    #
    # -  V1.4, Dec. 23 1997
    #   - Minor clean up.
    #   - Added a verbose switch
    #   - Added the sorting of the rows of B in order to fix in some reasonable way the
    #     permutation indetermination.  See note 2) below.
    #
    # -  V1.3, Nov.  2 1997
    #   - Some clean up.  Released in the public domain.
    #
    # -  V1.2, Oct.  5 1997
    #   - Changed random picking of the cumulant matrix used for initialization to a
    #     deterministic choice.  This is not because of a better rationale but to make
    #     the ouput (almost surely) deterministic.
    #   - Rewrote the joint diag. to take more advantage of Matlab"s tricks.
    #   - Created more dummy variables to combat Matlab"s loose memory management.
    #
    # -  V1.1, Oct. 29 1997.
    #    Made the estimation of the cumulant matrices more regular. This also corrects a
    #    buglet...
    #
    # -  V1.0, Sept. 9 1997. Created.
    #
    # Main references:
    # @article{CS-iee-94,
    #  title 	= "Blind beamforming for non {G}aussian signals",
    #  author       = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
    #  HTML 	= "ftp://sig.enst.fr/pub/jfc/Papers/iee.ps.gz",
    #  journal      = "IEE Proceedings-F",
    #  month = dec, number = 6, pages = {362-370}, volume = 140, year = 1993}
    #
    #
    # @article{JADE:NC,
    #  author  = "Jean-Fran\c{c}ois Cardoso",
    #  journal = "Neural Computation",
    #  title   = "High-order contrasts for independent component analysis",
    #  HTML    = "http://www.tsi.enst.fr/~cardoso/Papers.PS/neuralcomp_2ppf.ps",
    #  year    = 1999, month =	jan,  volume =	 11,  number =	 1,  pages =  "157-192"}
    #
    #
    #  Notes:
    #  ======
    #
    #  Note 1) The original Jade algorithm/code deals with complex signals in Gaussian
    #  noise white and exploits an underlying assumption that the model of independent
    #  components actually holds.  This is a reasonable assumption when dealing with
    #  some narrowband signals.  In this context, one may i) seriously consider dealing
    #  precisely with the noise in the whitening process and ii) expect to use the small
    #  number of significant eigenmatrices to efficiently summarize all the 4th-order
    #  information.  All this is done in the JADE algorithm.
    #
    #  In *this* implementation, we deal with real-valued signals and we do NOT expect
    #  the ICA model to hold exactly.  Therefore, it is pointless to try to deal
    #  precisely with the additive noise and it is very unlikely that the cumulant
    #  tensor can be accurately summarized by its first n eigen-matrices.  Therefore,
    #  we consider the joint diagonalization of the *whole* set of eigen-matrices.
    #  However, in such a case, it is not necessary to compute the eigenmatrices at all
    #  because one may equivalently use `parallel slices" of the cumulant tensor.  This
    #  part (computing the eigen-matrices) of the computation can be saved: it suffices
    #  to jointly diagonalize a set of cumulant matrices.  Also, since we are dealing
    #  with reals signals, it becomes easier to exploit the symmetries of the cumulants
    #  to further reduce the number of matrices to be diagonalized. These
    #  considerations, together with other cheap tricks lead to this version of JADE
    #  which is optimized (again) to deal with real mixtures and to work `outside the
    #  model'.  As the original JADE algorithm, it works by minimizing a `good set' of
    #  cumulants.
    #
    #  Note 2) The rows of the separating matrix B are resorted in such a way that the
    #  columns of the corresponding mixing matrix A=pinv(B) are in decreasing order of
    #  (Euclidian) norm.  This is a simple, `almost canonical" way of fixing the
    #  indetermination of permutation.  It has the effect that the first rows of the
    #  recovered signals (ie the first rows of B*X) correspond to the most energetic
    #  *components*.  Recall however that the source signals in S=B*X have unit
    #  variance.  Therefore, when we say that the observations are unmixed in order of
    #  decreasing energy, this energetic signature is to be found as the norm of the
    #  columns of A=pinv(B) and not as the variances of the separated source signals.
    #
    #  Note 3) In experiments where JADE is run as B=jadeR(X,m) with m varying in range
    #  of values, it is nice to be able to test the stability of the decomposition.  In
    #  order to help in such a test, the rows of B can be sorted as described above. We
    #  have also decided to fix the sign of each row in some arbitrary but fixed way.
    #  The convention is that the first element of each row of B is positive.
    #
    #  Note 4) Contrary to many other ICA algorithms, JADE (or least this version) does
    #  not operate on the data themselves but on a statistic (the full set of 4th order
    #  cumulant). This is represented by the matrix CM below, whose size grows as m^2 x
    #  m^2 where m is the number of sources to be extracted (m could be much smaller
    #  than n).  As a consequence, (this version of) JADE will probably choke on a
    #  `large' number of sources. Here `large' depends mainly on the available memory
    #  and could be something like 40 or so.  One of these days, I will prepare a
    #  version of JADE taking the `data' option rather than the `statistic' option.

    # Notes on translation (GB):
    # =========================
    #
    # Note 1) The function jadeR is a relatively literal translation from the original
    # MATLABcode. I haven't really looked into optimizing it for NumPy. If you have any
    # time to look at this and good ideas, let me know.
    #
    # Note 2) A test module that compares NumPy output with Octave (MATLAB
    # clone) output of the original MATLAB script is available


class JadeICA(TransformerMixin, BaseEstimator):
    """Perform blind source separation using joint diagonalization."""

    def __init__(self, *, n_components: Optional[int] = None) -> None:
        """Perform a blind signal separation using joint diagonalization.

        Parameters
        ----------
        n_components : int, default=None
            Number of signals to extract. `None` assumes all components
        """
        super().__init__()

        self.n_components: Optional[int] = n_components
        self.mean_: NDArray[(Any, ...), Float]
        self.components_: NDArray[(Any, ...), Float]

    def fit(
        self,
        arr: NDArray[(Any, ...), Float],
        y: Optional[NDArray[(Any, ...), Float]] = None,
    ) -> JadeICA:
        """Calculate the unmixing matrix.

        Parameters
        ----------
        arr : NDArray
            mixed signal array
        y : NDArray, optional
            unused

        Returns
        -------
        self
        """
        self.mean_ = arr.mean(axis=0)
        self.components_ = _jade(arr, n_components=self.n_components)
        return self

    def transform(self, arr: NDArray[(Any, ...), Float]) -> NDArray[(Any, ...), Float]:
        """Project the unmixing matrix onto `arr` to give independent signals.

        Parameters
        ----------
        arr : NDArray
            Unmixed signals

        Returns
        -------
        NDArray
            Unmixed signal array
        """
        check_is_fitted(self)

        arr -= self.mean_
        signal: NDArray[(Any, ...), Float] = arr @ self.components_.T
        return signal

    def fit_transform(
        self,
        arr: NDArray[(Any, ...), Float],
        y: Optional[NDArray[(Any, ...), Float]] = None,
        **fit_params: str,
    ) -> NDArray[(Any, ...), Float]:
        """Determine the independent signals using joint diagonalization.

        Parameters
        ----------
        arr : NDArray
            Mixed signal array
        y : NDArray, optional
            unused
        fit_params : dict
            unused

        Returns
        -------
        NDArray
            Unmixed signal array
        """
        self.mean_ = arr.mean(axis=0)
        self.components_ = _jade(arr, n_components=self.n_components)

        arr -= self.mean_
        signal: NDArray[(Any, ...), Float] = arr @ self.components_.T
        return signal

    def inverse_transform(
        self, arr: NDArray[(Any, ...), Float]
    ) -> NDArray[(Any, ...), Float]:
        """Find the original mixed signals.

        Parameters
        ----------
        arr : NDArray
            Unmixed signal array

        Raises
        ------
        NotImplementedError
            If called because Jade does not provide inversion
        """
        raise NotImplementedError(
            "Inverse transformation is currently not implemented."
        )
