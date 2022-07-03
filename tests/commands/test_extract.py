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
"""Test extract CLI subcommand."""
import logging
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from pytest_console_scripts import ScriptRunner

from ..datafile import TOPWW, TRAJDIH, TRAJFILES, TRJWW

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
LOGGER = logging.getLogger(name="ambgen.commands.cmd_extract")

if not sys.warnoptions:
    import os
    import warnings

    warnings.simplefilter("default")  # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses


class TestExtract:
    """Run test for extract subcommand."""

    def test_help(self, script_runner: ScriptRunner) -> None:
        """Test help output.

        GIVEN the extract subcommand
        WHEN the help option is invoked
        THEN the help output should be displayed

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run(
            "qaa",
            "extract",
            "-h",
        )

        assert "Usage:" in result.stdout
        assert result.success

    def test_extract_coords(self, script_runner: ScriptRunner, tmp_path: Path) -> None:
        """Test extract subcommand for coordinates.

        GIVEN a trajectory file
        WHEN invoking the extract subcommand
        THEN a trajectory file will be written

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        # Paths
        saveDir = tmp_path / "qaa"
        saveDir.mkdir(exist_ok=True)
        logfile = saveDir / "log.txt"
        output = saveDir / "coordinates.npy"

        result = script_runner.run(
            "qaa",
            "extract",
            "-c",
            TRAJFILES,
            "-p",
            TOPWW,
            "-f",
            TRJWW,
            "-o",
            saveDir,
            "-l",
            logfile,
            "--debug",
        )
        universe = mda.Universe(TOPWW, TRJWW)
        n_atoms = universe.select_atoms("protein and name CA").n_atoms
        n_frames = universe.trajectory.n_frames
        size = n_frames * n_atoms * 3

        assert result.success
        assert logfile.exists()
        assert output.exists()

        data = np.memmap(output, mode="r", dtype=np.float_)
        assert data.size == size

    def test_extract_dihedrals(
        self, script_runner: ScriptRunner, tmp_path: Path
    ) -> None:
        """Test extract subcommand for dihedrals.

        GIVEN a trajectory file
        WHEN invoking the extract subcommand
        THEN a trajectory file will be written

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        tmp_path : Path
            Temporary directory
        """
        # Paths
        saveDir = tmp_path / "qaa"
        saveDir.mkdir(exist_ok=True)
        logfile = saveDir / "log.txt"
        output = saveDir / "dihedrals.npy"

        result = script_runner.run(
            "qaa",
            "extract",
            "-c",
            TRAJDIH,
            "-p",
            TOPWW,
            "-f",
            TRJWW,
            "-o",
            saveDir,
            "-l",
            logfile,
            "--debug",
        )
        universe = mda.Universe(TOPWW, TRJWW)
        n_residues = universe.select_atoms(
            "backbone and resnum 2:132"
        ).residues.n_residues
        n_frames = universe.trajectory.n_frames
        size = n_frames * n_residues * 4

        assert result.success
        assert logfile.exists()
        assert output.exists()

        data = np.memmap(output, mode="r", dtype=np.float_)
        assert data.size == size
