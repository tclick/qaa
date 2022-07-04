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
from dataclasses import is_dataclass
from typing import Any, Dict

import pytest

from qaa.libs.configparser import Config, configure, parse

from ..datafile import BAD_CONFIG, TRAJFILES, TRAJFORM


class TestConfig:
    """Test Config class."""

    @pytest.fixture
    def context(self) -> Config:
        """Create an mock click.Context object.

        Returns
        -------
        Config
            a mock click.Context object
        """
        config = Config()
        config.analysis = "coordinates"
        return config

    def test_config(self, context: Config) -> None:
        """Test the Config class.

        GIVEN a Config class
        WHEN the class is initialized
        THEN the object should be of type `dataclass`

        Parameters
        ----------
        context: Config
            mock object
        """
        assert is_dataclass(context)
        assert hasattr(context, "analysis")
        assert context.analysis == "coordinates"

    def test_update(self, context: Config) -> None:
        """Test functionality of update method.

        GIVEN a Config object
        WHEN a dict is included in the update method
        THEN the object should add the new attribute

        Parameters
        ----------
        context: Config
            mock object
        """
        kwargs = dict(debug=True, startres=1)
        context.update(**kwargs)

        assert hasattr(context, "analysis")
        assert hasattr(context, "debug")
        assert hasattr(context, "startres")
        assert context.debug
        assert context.startres == 1


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

        GIVEN a click.Context and a filename
        WHEN the configure function is called
        THEN context.default_map is populated with the YAML configuration

        Parameters
        ----------
        context: Config
            mock object
        """
        configure(context, [], TRAJFILES)

        assert len(context.default_map) > 0
        assert "trajfiles" in context.default_map
        assert context.default_map["trajfiles"] is not None

    def test_parse(self, context: Config) -> None:
        """Test the configuration parsing function.

        GIVEN a click.Context object and a configuration file
        WHEN the contents of the file with `trajform` defined is parsed
        THEN the `trajfiles` list is populated

        Parameters
        ----------
        context: Config
            mock object
        """
        configure(context, [], TRAJFORM)
        parser = Config()
        parser.update(**context.default_map)
        parse(parser)

        assert parser.trajfiles is not None
        assert "pnas2013-native-1-protein-010.dcd" in parser.trajfiles

    def test_bad_config(self, context: Dict[Any, Any]) -> None:
        """Test a configuration file that has both `trajform` and `trajfiles` defined.

        GIVEN a YAML configuration file
        WHEN both `trajform` and `trajfiles` are defined
        THEN a ValueError is raised

        Parameters
        ----------
        context: Config
            mock object
        """
        configure(context, [], BAD_CONFIG)
        parser = Config()
        parser.update(**context.default_map)

        with pytest.raises(ValueError):
            parse(parser)
