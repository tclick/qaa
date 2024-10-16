# --------------------------------------------------------------------------------------
#  Copyright (C) 2020—2021 by Timothy H. Click <tclick@okstate.edu>
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
"""Test cases for the __main__ module."""
import pytest
import qaa.__main__
import qaa.cli
from pytest_console_scripts import ScriptRunner
from qaa import create_logging_dict


class TestMain:
    """Test the main module."""

    def test_main_help(self, script_runner: ScriptRunner) -> None:
        """Test help option.

        GIVEN the main command-line interface
        WHEN the '-h' or '--help' argument is provided
        THEN the help screen should appear

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run(
            "qaa",
            "-h",
        )
        result2 = script_runner.run(
            "qaa",
            "--help",
        )

        assert "Usage:" in result.stdout
        assert result.success
        assert result.stdout == result2.stdout

    def test_main_version(self, script_runner: ScriptRunner) -> None:
        """Test version option.

        GIVEN the main command-line interface
        WHEN the '--version' argument is provided
        THEN the version should print to the screen

        Parameters
        ----------
        script_runner : ScriptRunner
            Command-line runner
        """
        result = script_runner.run("qaa", "--version")
        assert qaa.__version__ in result.stdout


class TestLoggingDict:
    """Test creation of logging dictionary."""

    def test_create_logging_dict(self) -> None:
        """Check create_logging_dict return."""
        logfile = "test.log"
        assert isinstance(create_logging_dict(logfile), dict)

    def test_create_logging_dict_error(self) -> None:
        """Test whether an error is raised.

        GIVEN create_logging_dict function
        WHEN an empty string for a filename is provided
        THEN an exception is thrown
        """
        logfile = ""
        with pytest.raises(ValueError):
            create_logging_dict(logfile)
