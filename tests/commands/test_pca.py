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
"""Test CLI with pca subcommand."""
import logging
import sys
from pathlib import Path

import pytest
from pytest_console_scripts import ScriptRunner
from pytest_mock import MockerFixture

from ..datafile import TOPWW
from ..datafile import TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_pca")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestPCA:
    """Run test for pca subcommand."""

    @pytest.mark.runner_setup
    def test_help(self, script_runner: ScriptRunner) -> None:
        """Test help output.

        GIVEN the pca subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run(
            "qaa",
            "pca",
            "-h",
        )

        assert "Usage:" in result.stdout
        assert result.success

    @pytest.mark.runner_setup
    def test_pca(
        self, script_runner: ScriptRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test pca subcommand.

        GIVEN a trajectory file
        WHEN invoking the pca subcommand
        THEN an several files will be written

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        logfile = tmp_path.joinpath("pca.log")
        result = script_runner.run(
            "qaa",
            "pca",
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
            "--verbose",
        )

        assert result.success
        assert logfile.exists()
        assert tmp_path.joinpath("projection.csv").exists()
        assert not tmp_path.joinpath("explained_variance_ratio.png").exists()

    @pytest.mark.runner_setup
    def test_pca_with_image(
        self, script_runner: ScriptRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test pca subcommand with image option.

        GIVEN a trajectory file
        WHEN invoking the pca subcommand with an image option
        THEN an several files will be written including an image file

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        logfile = tmp_path.joinpath("pca.log")
        result = script_runner.run(
            "qaa",
            "pca",
            "-s",
            TOPWW,
            "-f",
            TRJWW,
            "-o",
            tmp_path.as_posix(),
            "-l",
            logfile.as_posix(),
            "--bias",
            "--whiten",
            "-m",
            "ca",
            "--image",
        )

        assert result.success
        assert logfile.exists()
        assert tmp_path.joinpath("projection.csv").exists()
