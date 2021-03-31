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
"""Test cluster subcommand"""
import logging
import sys

import pytest
from click.testing import CliRunner
from numpy import random
from numpy.typing import ArrayLike

from ..datafile import PROJ, TOPWW, TRJWW
from qaa.cli import main

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_qaa")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestCluster:
    @pytest.fixture
    def n_modes(self) -> int:
        return 5

    @pytest.fixture
    def data(self, n_modes: int) -> ArrayLike:
        rng = random.default_rng()
        return rng.standard_normal((10, n_modes))

    @pytest.mark.runner_setup
    def test_help(self, cli_runner: CliRunner):
        """
        GIVEN the qaa subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed
        """
        result = cli_runner.invoke(
            main,
            args=(
                "cluster",
                "-h",
            ),
        )

        assert "Usage:" in result.output
        assert result.exit_code == 0

    @pytest.mark.runner_setup
    def test_cluster(self, cli_runner: CliRunner, tmp_path, mocker):
        """
        GIVEN a data file
        WHEN invoking the cluster subcommand
        THEN saves a cluster image to disk
        """
        outfile = tmp_path.joinpath("cluster.png")
        logfile = outfile.with_suffix(".log")
        fig = mocker.patch("matplotlib.figure.Figure.savefig")
        result = cli_runner.invoke(
            main,
            args=(
                "cluster",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-i",
                PROJ,
                "-o",
                outfile,
                "-l",
                logfile,
                "--verbose",
            ),
        )
        assert result.exit_code == 0
        assert logfile.exists()
        fig.assert_called_once()
