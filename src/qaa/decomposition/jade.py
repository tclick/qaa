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
from typing import Optional

from numpy import abs
from numpy import append
from numpy import arange
from numpy import arctan2
from numpy import argsort
from numpy import array
from numpy import concatenate
from numpy import cos
from numpy import diag
from numpy import dot
from numpy import eye
from numpy import float64
from numpy import matrix
from numpy import multiply
from numpy import sign
from numpy import sin
from numpy import sqrt
from numpy import zeros
from numpy.linalg import eig
from numpy.linalg import pinv
from sklearn.decomposition import _base
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from ..libs.typing import ArrayType

logger = logging.getLogger(__name__)


def _jade(arr: ArrayType, m: Optional[int] = None, verbose: bool = True):
    """Blind separation of real signals with JADE.

    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm developed
    by Jean-Francois Cardoso. See http://www.tsi.enst.fr/~cardoso/guidesepsou.html, and
    papers cited at the end of the source file.

    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by Gabriel
    Beckers, http://gbeckers.nl .

    Parameters
    ----------
    arr : ArrayType
        a data matrix (n_features, n_samples)
    m : int, default=None
        output matrix B has size mxn so that only m sources are extracted.  This is done
        by restricting the operation of jadeR to the m first principal components.
        Defaults to None, in which case :math:`m = n`.
    verbose : bool
        print info on progress. Default is True.

    Returns
    -------
    ArrayType
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
    arr: ArrayType = check_array(arr)
    origtype = arr.dtype  # remember to return matrix B of the same type
    arr = matrix(arr.astype(float64))

    # GB: n_features is number of input signals, n_samples is number of samples
    n_features, n_samples = arr.shape

    if m is None:
        m = n_features  # Number of sources defaults to # of sensors
    if m > n_features:
        raise IndexError(f"More sources ({m}) than sensors ({n_features})")

    logger.info("jade -> Looking for %d sources", m)
    logger.info("jade -> Removing the mean value")
    arr -= arr.mean(axis=1)

    # whitening & projection onto signal subspace
    # ===========================================
    logger.info("jade -> Whitening the data")
    # An eigen basis for the sample covariance matrix
    eigenval, eigenvec = eig((arr * arr.T) / n_samples)
    k = eigenval.argsort()
    sorted_eval = eigenval[k]  # Sort by increasing variances
    pcs = arange(
        n_features - 1, n_features - m - 1, -1
    )  # The m most significant princip. comp. by decreasing variance

    # --- PCA  ----------------------------------------------------------
    components = eigenvec[
        :, k[pcs]
    ].T  # % At this stage, B does the PCA on m components

    # --- Scaling  ------------------------------------------------------
    scales = sqrt(sorted_eval[pcs])  # The scales of the principal components .
    # Now, B does PCA followed by a rescaling = sphering
    components = diag(1.0 / scales) * components
    # B[-1,:] = -B[-1,:] # GB: to make it compatible with octave
    # --- Sphering ------------------------------------------------------
    arr = (
        components * arr
    )  # %% We have done the easy part: B is a whitening matrix and X is white.

    del eigenvec, eigenval, sorted_eval, k, pcs, scales

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

    # Reshaping of the data, hoping to speed up things a little bit...
    arr = arr.T
    dimsymm = int((m * (m + 1)) / 2)  # Dim. of the space of real symm matrices
    nbcm = dimsymm  # number of cumulant matrices
    cm = matrix(zeros([m, m * nbcm], dtype=float64))  # Storage for cumulant matrices
    r = matrix(eye(m, dtype=float64))
    q_ij = matrix(zeros([m, m], dtype=float64))  # Temp for a cum. matrix
    xim = zeros(m, dtype=float64)  # Temp
    xijm = zeros(m, dtype=float64)  # Temp
    # Uns = numpy.ones([1,m], dtype=numpy.uint32)    # for convenience
    # GB: we don't translate that one because NumPy doesn't need Tony's rule

    # I am using a symmetry trick to save storage.  I should write a short note one of
    # these days explaining what is going on here.
    range_ = arange(m)  # will index the columns of CM where to store the cum. mats.

    for im in range(m):
        xim = arr[:, im]
        xijm = multiply(xim, xim)
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        q_ij = (
            multiply(xijm, arr).T * arr / float(n_samples)
            - r
            - 2 * dot(r[:, im], r[:, im].T)
        )
        cm[:, range_] = q_ij
        range_ = range_ + m
        for jm in range(im):
            xijm = multiply(xim, arr[:, jm])
            q_ij = (
                sqrt(2) * multiply(xijm, arr).T * arr / float(n_samples)
                - r[:, im] * r[:, jm].T
                - r[:, jm] * r[:, im].T
            )
            cm[:, range_] = q_ij
            range_ = range_ + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.
    vec_ = matrix(eye(m, dtype=float64))

    diag_ = zeros(m, dtype=float64)
    on = 0.0
    range_ = arange(m)
    for _ in range(nbcm):
        diag_ = diag(cm[:, range_])
        on = on + (diag_ * diag_).sum(axis=0)
        range_ = range_ + m
    off = (multiply(cm, cm).sum(axis=0)).sum(axis=0) - on

    seuil = 1.0e-6 / sqrt(
        n_samples
    )  # % A statistically scaled threshold on `small" angles
    encore = True
    sweep = 0  # % sweep number
    updates = 0  # % Total number of rotations
    upds = 0  # % Number of rotations in a given seep
    g = zeros([2, nbcm], dtype=float64)
    gg = zeros([2, 2], dtype=float64)
    g = zeros([2, 2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    gain = 0

    # Joint diagonalization proper
    logger.info("jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        logger.info("jade -> Sweep #%3d", sweep)
        sweep = sweep + 1
        upds = 0
        # Vkeep = V

        for p in range(m - 1):
            for q in range(p + 1, m):

                ip = arange(p, m * nbcm, m)
                iq = arange(q, m * nbcm, m)

                # computation of Givens angle
                g = concatenate([cm[p, ip] - cm[q, iq], cm[p, iq] + cm[q, ip]])
                gg = dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    g = matrix([[c, -s], [s, c]])
                    pair = array([p, q])
                    vec_[:, pair] = vec_[:, pair] * g
                    cm[pair, :] = g.T * cm[pair, :]
                    cm[:, concatenate([ip, iq])] = append(
                        c * cm[:, ip] + s * cm[:, iq],
                        -s * cm[:, ip] + c * cm[:, iq],
                        axis=1,
                    )
                    on = on + gain
                    off = off - gain

        logger.info("completed in %d rotations", upds)
        updates = updates + upds
    logger.info("jade -> Total of %d Givens rotations", updates)

    # A separating matrix
    # ===================

    components = vec_.T * components

    # Permute the rows of the separating matrix B to get the most energetic components
    # first. Here the **signals** are normalized to unit variance.  Therefore, the sort
    # is according to the norm of the columns of A = pinv(B)

    logger.info("jade -> Sorting the components")

    unmixing = pinv(components)
    keys = array(argsort(multiply(unmixing, unmixing).sum(axis=0)[0]))[0]
    components = components[keys, :]
    components = components[::-1, :]  # % Is this smart ?

    logger.info("jade -> Fixing the signs")
    b = components[:, 0]
    signs = array(sign(sign(b) + 0.1).T)[0]  # just a trick to deal with sign=0
    components = diag(signs) * components

    return array(components.astype(origtype))

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


class JadeICA(_base._BasePCA):
    """Perform blind source separation using joint diagonalization."""

    def __init__(self, *, n_components=None, verbose=True):
        """Perform a blind signal separation using joint diagonalization.

        Parameters
        ----------
        n_components : int, default=None
            Number of signals to extract. `None` assumes all components
        verbose : bool, default=True
            Display information during the calculation
        """
        super().__init__()

        self.n_components = n_components
        self.verbose = verbose
        self.mean_ = None
        self.components_ = None

    def fit(self, arr: ArrayType, y: Optional[ArrayType] = None) -> JadeICA:
        """Calculate the unmixing matrix.

        Parameters
        ----------
        arr : ArrayType
            mixed signal array
        y : ArrayType
            unused

        Returns
        -------
        self
        """
        self.mean_ = arr.mean(axis=0)
        self.components_ = _jade(arr.T, m=self.n_components, verbose=self.verbose)
        return self

    def transform(self, arr: ArrayType) -> ArrayType:
        """Project the unmixing matrix onto `arr` to give independent signals.

        Parameters
        ----------
        arr : ArrayType
            Unmixed signals

        Returns
        -------
        ArrayType
            Unmixed signal array
        """
        check_is_fitted(self)

        arr -= self.mean_
        signal = self.components_ @ arr.T
        return signal.T

    def fit_transform(
        self, arr: ArrayType, y: Optional[ArrayType] = None, **fit_params
    ) -> ArrayType:
        """Determine the independent signals using joint diagonalization.

        Parameters
        ----------
        arr : ArrayType
            Mixed signal array
        y : ArrayType
            unused
        fit_params : dict
            unused

        Returns
        -------
        ArrayType
            Unmixed signal array
        """
        self.mean_ = arr.mean(axis=0)
        self.components_ = _jade(arr.T, m=self.n_components, verbose=self.verbose)

        arr -= self.mean_
        signal = self.components_ @ arr.T
        return signal.T

    def inverse_transform(self, arr: ArrayType) -> ArrayType:
        """Find the original mixed signals.

        Parameters
        ----------
        arr : ArrayType
            Unmixed signal array

        Raises
        ------
        NotImplementedError
            If called because Jade does not provide inversion
        """
        raise NotImplementedError(
            "Inverse transformation is currently not implemented."
        )
