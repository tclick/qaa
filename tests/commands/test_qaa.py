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
"""Test jade CLI subcommand"""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from qaa.cli import main

from ..datafile import TOPWW, TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_qaa")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class Testqaa:
    @pytest.mark.runner_setup
    def test_help(self, cli_runner: CliRunner, tmp_path: Path):
        """
        GIVEN the qaa subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed
        """
        result = cli_runner.invoke(
            main,
            args=(
                "qaa",
                "-h",
            ),
        )

        assert "Usage:" in result.output
        assert result.exit_code == 0

    @pytest.mark.runner_setup
    def test_qaa(self, cli_runner: CliRunner, tmp_path: Path, mocker):
        """
        GIVEN a trajectory file
        WHEN invoking the qaa subcommand
        THEN an several files will be written
        """
        logfile = tmp_path.joinpath("qaa.log")
        patch = mocker.patch.object(np, "savetxt", autospec=True)
        result = cli_runner.invoke(
            main,
            args=(
                "qaa",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-o",
                tmp_path,
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
        assert tmp_path.joinpath("projection.csv").exists()
        assert not tmp_path.joinpath("explained_variance_ratio.png").exists()

    @pytest.mark.runner_setup
    def test_qaa_error(self, cli_runner: CliRunner, tmp_path: Path, mocker):
        """
        GIVEN stop < start
        WHEN invoking the qaa subcommand
        THEN exit code > 0
        """
        logfile = tmp_path.joinpath("qaa.log")
        result = cli_runner.invoke(
            main,
            args=(
                "qaa",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-o",
                tmp_path,
                "-l",
                logfile,
                "-b",
                "3",
                "-e",
                "1",
                "-m",
                "ca",
                "--verbose"
            ),
        )
        assert result.exit_code > 0

    @pytest.mark.runner_setup
    def test_qaa_with_image(self, cli_runner: CliRunner, tmp_path: Path, mocker):
        """
        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with an image option
        THEN an several files will be written
        """
        logfile = tmp_path.joinpath("qaa.log")
        patch = mocker.patch.object(np, "savetxt", autospec=True)
        fig = mocker.patch("matplotlib.figure.Figure.savefig")
        result = cli_runner.invoke(
            main,
            args=(
                "qaa",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-o",
                tmp_path,
                "-l",
                logfile,
                "--bias",
                "--whiten",
                "-m",
                "ca",
                "--image",
            ),
        )

        assert result.exit_code == 0
        patch.assert_called()
        fig.assert_called()
        assert logfile.exists()
        assert tmp_path.joinpath("projection.csv").exists()
        assert tmp_path.joinpath("explained_variance_ratio.png").exists()