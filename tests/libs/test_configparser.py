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
from typing import Any, Dict

import pytest

from qaa.libs.configparser import Config, configure, parse

from ..datafile import BAD_CONFIG, TRAJFILES, TRAJFORM


class TestConfigParser:
    """Test ConfigParser class."""

    @pytest.fixture
    def context(self) -> Config:
        """Create an mock click.Context object.

        Returns
        -------
        Config
            a mock click.Context object
        """
        config = Config()
        config.default_map = {}
        return config

    def test_configure(self, context: Config) -> None:
        """Test the configure function.

        Parameters
        ----------
        context: Config
            mock object

        GIVEN a click.Context and a filename
        WHEN the configure function is called
        THEN context.default_map is populated with the YAML configuration
        """
        configure(context, [], TRAJFILES)

        assert len(context.default_map) > 0
        assert "trajfiles" in context.default_map
        assert context.default_map["trajfiles"] is not None

    def test_parse(self, context: Config) -> None:
        """Test the configuration parsing function.

        Parameters
        ----------
        context: Config
            mock object

        GIVEN a click.Context object and a configuration file
        WHEN the contents of the file with `trajform` defined is parsed
        THEN the `trajfiles` list is populated
        """
        configure(context, [], TRAJFORM)
        context.default_map["trajfiles"] = None
        parser = Config(context.default_map)
        parse(parser)

        assert len(parser) > 0
        assert "pnas2013-native-1-protein-010.dcd" in parser.trajfiles

    def test_bad_config(self, context: Dict[Any, Any]) -> None:
        """Test a configuration file that has both `trajform` and `trajfiles` defined.

        GIVEN a YAML configuration file
        WHEN both `trajform` and `trajfiles` are defined
        THEN a ValueError is raised
        """
        configure(context, [], BAD_CONFIG)
        parser = Config(context.default_map)

        with pytest.raises(ValueError):
            parse(parser)
