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
"""Quasi-Anharmonic Analysis."""
import logging
import os
from typing import Any
from typing import Dict
from typing import TYPE_CHECKING
from typing import TypeVar

logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    PathLike = TypeVar("PathLike", str, os.PathLike[str])
else:
    PathLike = TypeVar("PathLike", str, os.PathLike)

__version__: str = "1.0.0-rc1"

_MASK: Dict[str, str] = dict(
    ca="protein and name CA",
    cab="protein and name =~ 'C[AB]'",
    back="protein and backbone",
    side="protein and not backbone and not (element =~ 'H')",
    noh="protein and not (element =~ 'H')",
    all="all",
)


def create_logging_dict(logfile: PathLike) -> Dict[str, Any]:
    """Configure the logger.

    Parameters
    ----------
    logfile : PathLike
        Filename for log output.

    Returns
    -------
    Dict
        Configuration data for logging.

    Raises
    ------
    ValueError
        If a filename is not defined.
    """
    if not logfile:
        raise ValueError("Filename not defined.")

    logger_dict = dict(
        version=1,
        disable_existing_loggers=False,  # this fixes the problem
        formatters=dict(
            standard={
                "class": "logging.Formatter",
                "format": "%(name)-12s %(levelname)-8s %(message)s",
            },
            detailed={
                "class": "logging.Formatter",
                "format": ("%(asctime)s %(name)-15s %(levelname)-8s " "%(message)s"),
                "datefmt": "%m-%d-%y %H:%M",
            },
        ),
        handlers=dict(
            console={
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
            },
            file={
                "class": "logging.FileHandler",
                "filename": logfile,
                "level": "INFO",
                "mode": "w",
                "formatter": "detailed",
            },
        ),
        root=dict(level="INFO", handlers=["console", "file"]),
    )
    return logger_dict
