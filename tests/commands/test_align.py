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
import logging
import sys
from pathlib import Path

import MDAnalysis as mda
import pytest
from click.testing import CliRunner

from qaa.cli import main

from ..datafile import TOPWW, TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_align")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestAlign:
    @pytest.mark.runner_setup
    def test_help(self, cli_runner: CliRunner, tmp_path: Path):
        """
        GIVEN the align subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed
        """
        result = cli_runner.invoke(
            main,
            args=(
                "align",
                "-h",
            ),
            env=dict(AMBERHOME=tmp_path.as_posix()),
        )

        assert "Usage:" in result.output
        assert result.exit_code == 0

    @pytest.mark.runner_setup
    def test_align(self, cli_runner: CliRunner, tmp_path: Path, mocker):
        """
        GIVEN a trajectory file
        WHEN invoking the align subcommand
        THEN an aligned trajectory and average structure file will be written
        """
        logfile = tmp_path.joinpath("align.log")
        patch = mocker.patch.object(mda, "Writer", autospec=True)
        result = cli_runner.invoke(
            main,
            args=(
                "align",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-r",
                tmp_path.joinpath("average.pdb"),
                "-o",
                tmp_path.joinpath("align.nc"),
                "-l",
                logfile,
                "-m",
                "ca",
            ),
        )

        assert result.exit_code == 0
        patch.assert_called()
        assert logfile.exists()