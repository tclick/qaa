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
"""Test jade CLI subcommand."""
import logging
import sys
from pathlib import Path

import pytest
from pytest_console_scripts import ScriptRunner

from ..datafile import TOPWW
from ..datafile import TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_qaa")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestQaa:
    """Run test for qaa subcommand."""

    @pytest.fixture
    def n_modes(self) -> int:
        """Return number of modes to compute.

        Returns
        -------
        int
            Number of components
        """
        return 9

    def test_help(self, script_runner: ScriptRunner) -> None:
        """Test help output.

        GIVEN the qaa subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run(
            "qaa",
            "qaa",
            "-h",
        )

        assert "Usage:" in result.stdout
        assert result.success

    def test_qaa_jade(
        self,
        script_runner: ScriptRunner,
        tmp_path: Path,
        n_modes: int,
    ) -> None:
        """Test qaa subcommand using JADE ICA.

        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with the --jade flag
        THEN output `signal.csv`

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        n_modes : int
            Number of components
        """
        logfile = tmp_path.joinpath("qaa.log")
        script_runner.run(
            "qaa",
            "qaa",
            "-s",
            TOPWW,
            "-f",
            TRJWW,
            "-o",
            tmp_path.as_posix(),
            "-l",
            logfile.as_posix(),
            "-m",
            "ca",
            "--jade",
            "-n",
            str(n_modes),
            "--verbose",
        )

        assert logfile.exists()
        assert tmp_path.joinpath("qaa-signals.csv").exists()
        assert not tmp_path.joinpath("qaa.png").exists()

    def test_qaa_fastica(
        self,
        script_runner: ScriptRunner,
        tmp_path: Path,
        n_modes: int,
    ) -> None:
        """Test qaa subcommand using FastICA.

        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with the --jade flag
        THEN output `signal.csv`

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        n_modes : int
            Number of components
        """
        logfile = tmp_path.joinpath("qaa.log")
        result = script_runner.run(
            "qaa",
            "qaa",
            "-s",
            TOPWW,
            "-f",
            TRJWW,
            "-o",
            tmp_path.as_posix(),
            "-l",
            logfile.as_posix(),
            "-m",
            "ca",
            "--fastica",
            "-n",
            str(n_modes),
            "-w",
            "--iter",
            "1000",
            "--tol",
            "0.01",
            "--verbose",
        )

        assert result.success
        assert logfile.exists()

        # Test whether text data file exists
        data = tmp_path.joinpath("qaa-signals.csv")
        assert data.exists()
        assert data.stat().st_size > 0

        # Test whether binary data file exists
        bindata = tmp_path.joinpath("qaa-signals.npy")
        assert bindata.exists()
        assert bindata.stat().st_size > 0

        assert not tmp_path.joinpath("qaa.png").exists()

    def test_qaa_with_image(
        self,
        script_runner: ScriptRunner,
        tmp_path: Path,
        n_modes: int,
    ) -> None:
        """Test qaa subcommand with image option.

        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with an image option
        THEN an several files will be written including an image file

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        n_modes : int
            Number of components
        """
        logfile = tmp_path.joinpath("qaa.log")
        result = script_runner.run(
            "qaa",
            "qaa",
            "-s",
            TOPWW,
            "-f",
            TRJWW,
            "-o",
            tmp_path.as_posix(),
            "-l",
            logfile.as_posix(),
            "-m",
            "ca",
            "--jade",
            "-n",
            str(n_modes),
            "--image",
            "--verbose",
        )

        assert result.success
        assert logfile.exists()

        image = tmp_path.joinpath("qaa.png")
        assert image.exists()
        assert image.stat().st_size > 0
