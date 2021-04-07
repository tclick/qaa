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

import numpy as np
import pytest
from click.testing import CliRunner
from numpy import random
from numpy.typing import ArrayLike
from pytest_mock import MockerFixture
from qaa.cli import main
from qaa.decomposition.jade import JadeICA
from sklearn.decomposition import FastICA

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
        return 5

    @pytest.fixture
    def data(self, n_modes: int) -> ArrayLike:
        """Generate random data.

        Parameters
        ----------
        n_modes : int
            Number of components

        Returns
        -------
        ArrayLike
            Randomly generated array (n_features, n_components)
        """
        rng = random.default_rng()
        return rng.standard_normal((10, n_modes))

    @pytest.mark.runner_setup
    def test_help(self, cli_runner: CliRunner) -> None:
        """Test help output.

        GIVEN the qaa subcommand
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
                "qaa",
                "-h",
            ],
        )

        assert "Usage:" in result.output
        assert result.exit_code == 0

    @pytest.mark.runner_setup
    def test_qaa_jade(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mocker: MockerFixture,
        data: ArrayLike,
        n_modes: int,
    ) -> None:
        """Test qaa subcommand using JADE ICA.

        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with the --jade flag
        THEN output `signal.csv`

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        data : ArrayLike
            Data to represent ICA calculation
        n_modes : int
            Number of components
        """
        logfile = tmp_path.joinpath("qaa.log")
        ica = mocker.patch.object(JadeICA, "fit_transform", return_value=data)
        save_txt = mocker.patch.object(np, "savetxt", autospec=True)
        save = mocker.patch.object(np, "save", autospec=True)
        result = cli_runner.invoke(
            main,
            args=[
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
            ],
        )

        assert result.exit_code == 0
        ica.assert_called_once()
        save_txt.assert_called()
        save.assert_called()
        assert logfile.exists()
        assert tmp_path.joinpath("qaa-signals.csv").exists()
        assert not tmp_path.joinpath("qaa.png").exists()

    @pytest.mark.runner_setup
    def test_qaa_fastica(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mocker: MockerFixture,
        data: ArrayLike,
        n_modes: int,
    ) -> None:
        """Test qaa subcommand using FastICA.

        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with the --jade flag
        THEN output `signal.csv`

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        data : ArrayLike
            Data to represent ICA calculation
        n_modes : int
            Number of components
        """
        logfile = tmp_path.joinpath("qaa.log")
        ica = mocker.patch.object(FastICA, "fit_transform", return_value=data)
        save_txt = mocker.patch.object(np, "savetxt", autospec=True)
        save = mocker.patch.object(np, "save", autospec=True)
        cli_runner.invoke(
            main,
            args=[
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
            ],
        )

        ica.assert_called_once()
        save_txt.assert_called()
        save.assert_called()
        assert logfile.exists()
        assert tmp_path.joinpath("qaa-signals.csv").exists()
        assert not tmp_path.joinpath("qaa.png").exists()

    @pytest.mark.runner_setup
    def test_qaa_with_image(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        mocker: MockerFixture,
        data: ArrayLike,
        n_modes: int,
    ) -> None:
        """Test qaa subcommand with image option.

        GIVEN a trajectory file
        WHEN invoking the qaa subcommand with an image option
        THEN an several files will be written including an image file

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        mocker : MockerFixture
            Mock object
        data : ArrayLike
            Data to represent ICA calculation
        n_modes : int
            Number of components
        """
        logfile = tmp_path.joinpath("qaa.log")
        ica = mocker.patch.object(JadeICA, "fit_transform", return_value=data)
        save_txt = mocker.patch.object(np, "savetxt", autospec=True)
        save = mocker.patch.object(np, "save", autospec=True)
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
            ),
        )

        assert result.exit_code == 0
        ica.assert_called_once()
        save_txt.assert_called()
        save.assert_called()
        fig.assert_called_once()
        assert logfile.exists()
        assert tmp_path.joinpath("qaa-signals.csv").exists()
        assert tmp_path.joinpath("qaa.png").exists()
