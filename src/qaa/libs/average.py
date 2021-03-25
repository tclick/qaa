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
"""Determine the average structure of a trajectory."""
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

from .typing import ArrayType
from .typing import AtomType


class AverageStructure(AnalysisBase):
    def __init__(self, atomgroup, **kwargs):
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        self._ag: AtomType = atomgroup
        self.n_frames_: int = atomgroup.universe.trajectory.n_frames

    def _prepare(self):
        self.positions: ArrayType = np.zeros_like(self._ag.positions)

    def _single_frame(self):
        self.positions += self._ag.positions

    def _conclude(self):
        self.positions /= self.n_frames_
