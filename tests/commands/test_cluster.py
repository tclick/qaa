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
"""Test cluster subcommand."""
import logging
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture
from qaa.cli import main

from ..datafile import PROJ
from ..datafile import PROJNP
from ..datafile import TOPWW
from ..datafile import TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_qaa")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestCluster:
    """Run test for cluster subcommand."""

    @pytest.fixture
    def n_modes(self) -> int:
        """Return number of modes to compute.

        Returns
        -------
        int
            Number of components
        """
        return 5

    @pytest.mark.runner_setup
    def test_help(self, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the cluster subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(
            main,
            args=[
                "cluster",
                "-h",
            ],
        )

        assert "Usage:" in result.output
        assert result.exit_code == 0

    @pytest.mark.runner_setup
    def test_cluster_csv(
        self, cli_runner: CliRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test cluster subcommand with CSV input file.

        GIVEN a data file
        WHEN invoking the cluster subcommand
        THEN saves a cluster image to disk

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        outfile = tmp_path.joinpath("cluster.png")
        logfile = outfile.with_suffix(".log")
        fig = mocker.patch("matplotlib.figure.Figure.savefig")
        result = cli_runner.invoke(
            main,
            args=[
                "cluster",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-i",
                PROJ,
                "-o",
                outfile.as_posix(),
                "-l",
                logfile.as_posix(),
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        assert logfile.exists()
        fig.assert_called_once()

    @pytest.mark.runner_setup
    def test_cluster_npy(
        self, cli_runner: CliRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test cluster subcommand with binary Numpy input file.

        GIVEN a data file
        WHEN invoking the cluster subcommand
        THEN saves a cluster image to disk

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        outfile = tmp_path.joinpath("cluster.png")
        logfile = outfile.with_suffix(".log")
        fig = mocker.patch("matplotlib.figure.Figure.savefig")
        result = cli_runner.invoke(
            main,
            args=[
                "cluster",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-i",
                PROJNP,
                "-o",
                outfile.as_posix(),
                "-l",
                logfile.as_posix(),
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        assert logfile.exists()
        fig.assert_called_once()

    @pytest.mark.runner_setup
    def test_cluster_save(
        self, cli_runner: CliRunner, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test save option.

        GIVEN trajectory, topology and data files
        WHEN the '--save' option is provided
        THEN PDB files will be saved

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        """
        outfile = tmp_path.joinpath("cluster.png")
        logfile = outfile.with_suffix(".log")
        fig = mocker.patch("matplotlib.figure.Figure.savefig")
        pdb = mocker.patch("mdtraj.Trajectory.save")
        result = cli_runner.invoke(
            main,
            args=[
                "cluster",
                "-s",
                TOPWW,
                "-f",
                TRJWW,
                "-i",
                PROJ,
                "-o",
                outfile.as_posix(),
                "-l",
                logfile.as_posix(),
                "--save",
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        assert logfile.exists()
        fig.assert_called_once()
        pdb.assert_called()
