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
"""Various data files for testing."""
from pathlib import Path

from pkg_resources import resource_filename

__all__ = ["PROJ", "TOPWW", "TRJWW"]

PROJ = resource_filename(__name__, Path().joinpath("data", "projection.csv").as_posix())
PROJNP = resource_filename(
    __name__, Path().joinpath("data", "projection.npy").as_posix()
)
TOPWW = resource_filename(__name__, Path().joinpath("data", "protein.parm7").as_posix())
TRJWW = resource_filename(__name__, Path().joinpath("data", "protein.nc").as_posix())
