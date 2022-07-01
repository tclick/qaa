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
"""Functions to align coordinate files."""
import logging

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_squared_error

from .kabsch import Kabsch

logger = logging.getLogger(__name__)


def align_trajectory(
    mobile: npt.NDArray[np.float_],
    reference: npt.NDArray[np.float_],
    *,
    tol: float = 1e-6,
    verbose: bool = True,
) -> npt.NDArray[np.float_]:
    """Align `mobile` to `reference` using the Kabsch method.

    Parameters
    ----------
    mobile : array_like
        Positions of shape (n_frames, n_points, n_dims)
    reference : array_like
        Positions of shape (n_points, n_dims)
    tol : float
        Tolerance level for alignment
    verbose : bool
        Print alignment progress

    Returns
    -------
    array_like
        Aligned mobile array of shape (n_frames, n_points, n_dims)
    """
    error: float = np.inf
    iter: int = 0

    while error >= tol:
        kabsch = Kabsch(verbose=verbose)
        mobile[:] = np.asarray([kabsch.fit_transform(_, reference) for _ in mobile])
        new_reference: npt.NDArray[np.float_] = mobile.mean(axis=0)
        error = mean_squared_error(new_reference, reference, squared=False)
        reference = new_reference.copy()

        iter += 1
        if verbose:
            logger.info("Iteration %d -- error: %.6f", iter, error)

    if not verbose:
        logger.info("Total iterations: %d -- error: %.6f", iter, error)

    return mobile
