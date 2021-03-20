# --------------------------------------------------------------------------------------
#  Copyright (C) 2021 by Timothy H. Click <tclick@okstate.edu>
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
"""Annotation type definitions."""
from pathlib import Path
from typing import NewType, TypeVar, Union

import MDAnalysis as mda
from numpy.typing import ArrayLike
from pandas._typing import FrameOrSeries

PathLike = TypeVar("PathLike", str, Path)

# MDAnalysis types
AtomType = NewType("AtomType", mda.AtomGroup)
UniverseType = NewType("UniverseType", mda.Universe)
AtomUniv = Union[UniverseType, AtomType]
