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
import runpy

import pytest
import qaa.__main__
import qaa.cli
from click.testing import CliRunner
from qaa import create_logging_dict


class TestMain:
    """Test the main module."""

    def test_main_module(self) -> None:
        """Test the __main__ module.

        GIVEN the main command-line module
        WHEN the module is executed
        THEN the `qaa` module should be present
        """
        sys_dict = runpy.run_module("qaa.__main__")
        assert sys_dict["__name__"] == "qaa.__main__"
        assert isinstance(sys_dict["main"], qaa.cli._ComplexCLI)

    @pytest.mark.runner_setup
    def test_main_help(self, cli_runner: CliRunner) -> None:
        """Test help option.

        GIVEN the main command-line interface
        WHEN the '-h' or '--help' argument is provided
        THEN the help screen should appear

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(qaa.cli.main, args=("-h",))
        result2 = cli_runner.invoke(qaa.cli.main, args=("--help",))

        assert "Usage:" in result.output
        assert result.exit_code == 0
        assert result.output == result2.output

    @pytest.mark.runner_setup
    def test_main_version(self, cli_runner: CliRunner) -> None:
        """Test version option.

        GIVEN the main command-line interface
        WHEN the '--version' argument is provided
        THEN the version should print to the screen

        Parameters
        ----------
        cli_runner : CliRunner
            Command-line runner
        """
        result = cli_runner.invoke(qaa.cli.main, args=("--version",))
        assert qaa.__version__ in result.output


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
