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
from collections import Counter
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pylibyaml
import yaml

from .. import PathLike

logger: logging.Logger = logging.getLogger(__name__)


class ConfigParser:
    def __init__(self, filename: PathLike):
        """Initialize the config parser.

        Parameters
        ----------
        filename : PathLike
            name of configuration file
        """
        self._filename: PathLike = filename
        self._config: Dict[str, Any] = {}
        self.trajfiles: List[str] = []
        self.trajform: List[str] = []

    def load(self) -> None:
        with open(self._filename) as config_file:
            self._config = yaml.safe_load(config_file)
            if "_config" not in locals():
                IOError(
                    "Issue opening and reading configuration file: {!r}".format(
                        self._filename
                    )
                )

    def parse(self) -> None:
        """Parse the configuration file.

        Raises
        ------
        ValueError
            If both 'trajfiles' and 'trajform' are defined in the configuration file

        Notes
        -----
        The configuration file should look something like the following example:

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
        #trajform:
        #    - "pnas2013-native-1-protein-***.dcd"
        #    - "1-10"

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
        for key, value in self._config.items():
            setattr(self, key, value)

        if self.trajform and self.trajfiles:
            raise ValueError(
                "You cannot have both 'trajform' and 'trajfiles' "
                "defined in your configuration file. Please edit your file."
            )

        if self.trajform:
            filename = self.trajform[0]
            start, stop = self.trajform[1].split("-")
            values = Counter(filename)
            padval = values["*"]
            prefix, suffix = filename.split("*" * padval)
            self.trajfiles = [
                "{}{:0{}d}{}".format(prefix, _, padval, suffix)
                for _ in range(int(start), int(stop) + 1)
            ]