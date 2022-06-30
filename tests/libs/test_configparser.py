# --------------------------------------------------------------------------------------
#  Copyright (C) 2020–2022 by Timothy H. Click <Timothy.Click@briarcliff.edu>
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
"""Test ConfigParser class."""
import pytest

from qaa.libs.configparser import Config, ConfigParser

from ..datafile import BAD_CONFIG, TRAJFILES, TRAJFORM


class TestConfigParser:
    """Test ConfigParser class."""

    def test_load(self) -> None:
        """Test the load method.

        GIVEN a YAML filename
        WHEN a configuration is loaded
        THEN the parameters are added to the class
        """
        parser = ConfigParser(TRAJFILES)
        parser.load()
        assert isinstance(parser._config, Config)
        assert len(parser._config.keys()) > 0
        assert hasattr(parser._config, "trajfiles")

    def test_parse_trajfiles(self) -> None:
        """Test a configuration file with `trajfiles` set.

        GIVEN a YAML filename
        WHEN a configuration is loaded
        THEN the parameters are added to the class with a list of trajectory files
        """
        parser = ConfigParser(TRAJFILES)
        parser.load()
        config = parser.parse()
        assert len(config.trajfiles) == 1
        assert not hasattr(config, "trajform")

    def test_parse_trajform(self) -> None:
        """Test a configuration file with `trajform` set.

        GIVEN a YAML filename
        WHEN a configuration is loaded
        THEN the parameters are added to the class with a list of trajectory files
        """
        parser = ConfigParser(TRAJFORM)
        parser.load()
        config = parser.parse()
        assert len(config.trajform) == 2
        assert "pnas2013-native-1-protein-010.dcd" in config.trajfiles
        assert len(config.trajfiles) == 10

    def test_bad_config(self) -> None:
        """Test a configuration file that has both `trajform` and `trajfiles` defined.

        GIVEN a YAML configuration file
        WHEN both `trajform` and `trajfiles` are defined
        THEN a ValueError is raised
        """
        parser = ConfigParser(BAD_CONFIG)
        parser.load()
        with pytest.raises(ValueError):
            parser.parse()
