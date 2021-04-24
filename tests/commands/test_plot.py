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
"""Test plot subcommand."""
import logging
import sys
from pathlib import Path

from pytest_console_scripts import ScriptRunner

from ..datafile import CENTNPY
from ..datafile import CENTROID
from ..datafile import CLUSTER
from ..datafile import CLUSTNPY
from ..datafile import LABELS
from ..datafile import PROJ
from ..datafile import PROJNP

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(name="ambgen.commands.cmd_qaa")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestPlot:
    """Run test for cluster subcommand."""

    def test_help(self, script_runner: ScriptRunner) -> None:
        """Test help output.

        GIVEN the cluster subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run(
            "qaa",
            "plot",
            "-h",
        )
        assert result.success

    def test_plot_csv(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test plot subcommand with CSV input file.

        GIVEN a data file
        WHEN invoking the plot subcommand
        THEN saves a plotted image to disk

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        outfile = tmp_path.joinpath("projection.png")
        logfile = tmp_path.joinpath("plot.log")
        result = script_runner.run(
            "qaa",
            "plot",
            "-i",
            PROJ,
            "-o",
            outfile.as_posix(),
            "-l",
            logfile.as_posix(),
            "--pca",
            "--verbose",
        )
        assert result.success
        assert logfile.exists()
        assert outfile.exists()
        assert outfile.stat().st_size > 0

    def test_plot_npy(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test plot subcommand with NumPy input file.

        GIVEN a data file
        WHEN invoking the plot subcommand
        THEN saves a plotted image to disk

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        outfile = tmp_path.joinpath("projection.png")
        logfile = tmp_path.joinpath("plot.log")
        result = script_runner.run(
            "qaa",
            "plot",
            "-i",
            PROJNP,
            "-o",
            outfile.as_posix(),
            "-l",
            logfile.as_posix(),
            "--pca",
            "--verbose",
        )
        assert result.success
        assert logfile.exists()
        assert outfile.exists()
        assert outfile.stat().st_size > 0

    def test_plot_error(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test plot subcommand with non-existent file.

        GIVEN a non-existent data file
        WHEN invoking the plot subcommand
        THEN an error is thrown

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        outfile = tmp_path.joinpath("projection.png")
        logfile = tmp_path.joinpath("plot.log")
        result = script_runner.run(
            "qaa",
            "plot",
            "-i",
            "test.csv",
            "-o",
            outfile.as_posix(),
            "-l",
            logfile.as_posix(),
            "--pca",
            "--verbose",
        )
        assert not result.success
        assert not logfile.exists()
        assert not outfile.exists()

    def test_cluster_csv(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test plot subcommand with CSV input file.

        GIVEN a data file
        WHEN invoking the plot subcommand with cluster option
        THEN saves a plotted image to disk

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        outfile = tmp_path.joinpath("pca-cluster.png")
        logfile = tmp_path.joinpath("plot.log")
        result = script_runner.run(
            "qaa",
            "plot",
            "-i",
            CLUSTER,
            "-c",
            CENTROID,
            "-o",
            outfile.as_posix(),
            "-l",
            logfile.as_posix(),
            "--pca",
            "--cluster",
            "--verbose",
        )
        assert result.success
        assert logfile.exists()
        assert outfile.exists()
        assert outfile.stat().st_size > 0

    def test_cluster_npy(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test plot subcommand with binary NumPy input file.

        GIVEN a data file
        WHEN invoking the plot subcommand with cluster option
        THEN saves a plotted image to disk

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        outfile = tmp_path.joinpath("pca-cluster.png")
        logfile = tmp_path.joinpath("plot.log")
        result = script_runner.run(
            "qaa",
            "plot",
            "-i",
            CLUSTNPY,
            "-c",
            CENTNPY,
            "--label",
            LABELS,
            "-o",
            outfile.as_posix(),
            "-l",
            logfile.as_posix(),
            "--pca",
            "--cluster",
            "--verbose",
        )
        assert result.success
        assert logfile.exists()
        assert outfile.exists()
        assert outfile.stat().st_size > 0

    def test_no_cluster(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test plot subcommand with CSV input file.

        GIVEN a data file
        WHEN invoking the plot subcommand with cluster option
        THEN saves a plotted image to disk

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        outfile = tmp_path.joinpath("pca-cluster.png")
        logfile = tmp_path.joinpath("plot.log")
        result = script_runner.run(
            "qaa",
            "plot",
            "-i",
            CLUSTER,
            "-c",
            CENTROID,
            "-o",
            outfile.as_posix(),
            "-l",
            logfile.as_posix(),
            "--pca",
            "--verbose",
        )
        assert result.success
        assert logfile.exists()
        assert outfile.exists()
        assert outfile.stat().st_size > 0
