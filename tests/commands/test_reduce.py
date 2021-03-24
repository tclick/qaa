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

from pathlib import Path

import MDAnalysis as mda
import pytest
from click.testing import CliRunner

from qaa.cli import main

from ..datafile import TOPWW, TRJWW


class TestReduce:
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
                "reduce",
                "-h",
            ),
        )

        assert "Usage:" in result.output
        assert result.exit_code == 0

    @pytest.mark.runner_setup
    def test_reduce(self, cli_runner: CliRunner, tmp_path: Path, mocker):
        """
        GIVEN a trajectory file
        WHEN invoking the reduce subcommand
        THEN a trajectory is reduced to fewer atoms
        """
        logfile = tmp_path.joinpath("reduce.log")
        patch = mocker.patch.object(mda, "Writer", autospec=True)
        result = cli_runner.invoke(
            main,
            args=(
                "reduce",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-o",
                tmp_path.joinpath("reduce.nc"),
                "-l",
                logfile,
                "-m",
                "ca",
                "--verbose"
            ),
        )

        assert result.exit_code == 0
        patch.assert_called()
        assert logfile.exists()

    @pytest.mark.runner_setup
    def test_reduce_error(self, cli_runner: CliRunner, tmp_path: Path, mocker):
        """
        GIVEN stop < start
        WHEN invoking the reduce subcommand
        THEN exit code > 0
        """
        logfile = tmp_path.joinpath("reduce.log")
        result = cli_runner.invoke(
            main,
            args=(
                "reduce",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-o",
                tmp_path.joinpath("reduce.nc"),
                "-l",
                logfile,
                "-b",
                "3",
                "-e",
                "1",
                "-m",
                "ca",
            ),
        )
        assert result.exit_code > 0
