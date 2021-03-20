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

import MDAnalysis as mda
from numpy import testing

from qaa.libs import utils

from ..datafile import TOPWW, TRJWW


class TestUtils:
    def test_positions(self):
        """
        GIVEN topology and trajectory filenames
        WHEN the get_positions function is called
        THEN return an array of positions with shape (n_frames, n_atoms, 3)
        """
        universe = mda.Universe(TOPWW, TRJWW)
        n_atoms = universe.atoms.n_atoms
        n_frames = universe.trajectory.n_frames

        array = utils.get_positions(TOPWW, TRJWW)
        assert array.shape == (n_frames, n_atoms, 3)
        testing.assert_allclose(array[0], universe.atoms.positions)

        universe.trajectory[-1]
        testing.assert_allclose(array[-1], universe.atoms.positions)
