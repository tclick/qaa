# --------------------------------------------------------------------------------------
#  Copyright (C) 2020–2022 by Timothy H. Click <Timothy.Click@briarcliff.edu>
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
"""Parse a configuration file."""
import logging
from collections import Counter, UserDict
from pathlib import Path
from typing import Any, Dict, List

import click
import pylibyaml  # noqa: F401
import yaml

from .. import PathLike

logger: logging.Logger = logging.getLogger(__name__)


class Config(UserDict[str, Any]):
    """Configuration data."""

    def _validate_key(self, key: str) -> None:
        """Validate the configuration key.

        Parameters
        ----------
        key : str
            Configuration key

        Raises
        ------
        AttributeError
            If the key already exists
        ValueError
            If the key is invalid
        """
        if key in dir(self):
            raise AttributeError(f"'{key}' is a protected dictionary attribute")
        elif isinstance(key, str) and not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid attribute")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create a configuration class.

        Parameters
        ----------
        args
            Positional arguments
        kwargs
            Keyword arguments

        Raises
        ------
        AttributeError
            If `data` exists in the passed data
        """
        kwargs = dict(*args, **kwargs)
        if "data" in kwargs.keys():
            raise AttributeError("'data' is a protected dictionary attribute")
        self.__dict__["data"] = {}
        self.update(kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        """Set data according to a provided key.

        Parameters
        ----------
        key : str
            configuration key
        item : Any
            data
        """
        self._validate_key(key)
        super().__setitem__(key, item)

    def __setattr__(self, attr: str, val: Any) -> None:
        """Set attribute data.

        Parameters
        ----------
        attr : str
            attribute
        val : Any
            data
        """
        if attr == "data":
            super().__setattr__(attr, val)
        else:
            self.__setitem__(attr, val)

    def __getattr__(self, attr: str) -> Any:
        """Return data.

        Parameters
        ----------
        attr : str
            configuration key

        Returns
        -------
        Data

        Raises
        ------
        AttributeError
            If key does not exist
        """
        try:
            return self[attr]
        except KeyError as err:
            raise AttributeError(f"'Config' object has no attribute '{attr}'") from err

    def __delattr__(self, attr: str) -> None:
        """Remove the key and data.

        Parameters
        ----------
        attr : str
            configuration key

        Raises
        ------
        AttributeError
            If key does not exist
        """
        try:
            del self[attr]
        except KeyError as err:
            raise AttributeError(f"'Config' object has no attribute '{attr}'") from err

    def __getstate__(self) -> Dict[str, Any]:
        """Return the underlying data.

        Returns
        -------
        dictionary of underlying data
        """
        return self.data

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the configuration class.

        Parameters
        ----------
        state : Dict
            data
        """
        self.data = state


def configure(ctx: click.Context, param: List[Any], filename: PathLike) -> None:
    """Read a configuration file and pouplate click.Context.default_map with contents.

    Parameters
    ----------
    ctx : click.Context
        Context object
    param : list
        List of parameters
    filename : PathLike
        configuration file
    """
    if Path(filename).exists():
        logger.debug(f"Using configuration file: {filename}")
        with open(filename) as config_file:
            ctx.default_map = yaml.safe_load(config_file)


def parse(parser: Config) -> Config:
    """Parse the configuration file.

    Parameters
    ----------
    parser: Config
        configuration data

    Returns
    -------
    Config
        configuration data

    Raises
    ------
    ValueError
        If both 'trajfiles' and 'trajform' are defined in the configuration file

    Notes
    -----
    The configuration file should look something like the following example. Use
    either 'trajfiles' or 'trajform' but not both.

    analysis: "coordinate" # Can either be 'coordinate' or 'dihedral'
    verbose: False # Verbose output
    debug: False # Ouput debugging information
    graph: False # Save graphical data

    # Runs setup calculations: Cum. Sum. of cov. spectrum and unit radius neighbor search.
    setup: False

    # Topology and trajectory files
    topology: "pentapeptide/init-ww-penta.pdb"
    # Explicit list of trajectory files
    trajfiles:
      - "pentapeptide/job1-protein.dcd"
      - "pentapeptide/job2-protein.dcd"
    # Define trajectory filenames using regular expressions and zero-padded numbers
    trajform:
      - "pnas2013-native-1-protein-***.dcd" # noqa: RST210, RST213
      - "1-10"

    pname: "pentapeptide" # name of protein
    startRes: 1 # Starting residue number
    endRes: 5 # Final residue number
    icaDim: 8 # Number of dimensions calculated
    sliceVal: 1 # Number of frames to use

    # Locations of output data
    saveDir: "savefiles/" # Output subdirectory
    logfile: "log/log.txt"
    figDir: "savefiles/figures/" # Figure subdirectory if 'graph: True'
    """
    if parser.analysis != "coordinates" and parser.analysis != "dihedrals":
        raise ValueError("Analysis type must either be 'coordinates' or 'dihedrals'")
    if parser.trajform is not None and parser.trajfiles is not None:
        raise ValueError(
            "You cannot have both 'trajform' and 'trajfiles' "
            "defined in your configuration file. Please edit your file."
        )

    if parser.trajform is not None:
        filename = parser.trajform[0]
        start, stop = parser.trajform[1].split("-")
        padval = Counter(filename)["*"]
        prefix, suffix = filename.split("*" * padval)
        parser.trajfiles = [
            "{}{:0{}d}{}".format(prefix, _, padval, suffix)
            for _ in range(int(start), int(stop) + 1)
        ]

    return parser
