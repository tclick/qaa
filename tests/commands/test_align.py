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
"""Test align CLI subcommand."""
import logging
import sys
from pathlib import Path

from pytest_console_scripts import ScriptRunner
from pytest_mock import MockerFixture

from ..datafile import TOPWW
from ..datafile import TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_align")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestAlign:
    """Run test for align subcommand."""

    def test_help(self, script_runner: ScriptRunner) -> None:
        """Test help output.

        GIVEN the align subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run(
            "qaa",
            "align",
            "-h",
        )

        assert "Usage:" in result.stdout
        assert result.success

    def test_align(
        self, script_runner: ScriptRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test align subcommand.

        GIVEN a trajectory file
        WHEN invoking the align subcommand
        THEN an aligned trajectory and average structure file will be written

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        logfile = tmp_path.joinpath("align.log")
        result = script_runner.run(
            "qaa",
            "align",
            "-s",
            TOPWW,
            "-f",
            TRJWW,
            "-r",
            tmp_path.joinpath("average.pdb").as_posix(),
            "-o",
            tmp_path.joinpath("align.nc").as_posix(),
            "-l",
            logfile.as_posix(),
            "-m",
            "ca",
        )

        assert result.success
        assert logfile.exists()

    def test_align_verbose(
        self, script_runner: ScriptRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test align subcommand with verbose option.

        GIVEN a trajectory file
        WHEN invoking the align subcommand
        THEN an aligned trajectory and average structure file will be written

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        logfile = tmp_path.joinpath("align.log")
        result = script_runner.run(
            "qaa",
            "align",
            "-s",
            TOPWW,
            "-f",
            TRJWW,
            "-r",
            tmp_path.joinpath("average.pdb").as_posix(),
            "-o",
            tmp_path.joinpath("align.nc").as_posix(),
            "-l",
            logfile.as_posix(),
            "-m",
            "ca",
            "--verbose",
        )

        assert result.success
        assert logfile.exists()
