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
"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -mqaa` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``qaa.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``qaa.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import logging
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import click
from click import core

from . import __version__

CONTEXT_SETTINGS = dict(
    auto_envvar_prefix="COMPLEX", help_option_names=["-h", "--help"]
)
logger = logging.getLogger()


class _Context(click.Context):
    """Context manager for click command-line interface."""

    def __init__(self) -> None:
        self.verbose = False
        self.home = Path.home()

    def log(self, fmt: str, *messages: Tuple[str]) -> None:
        """Log a message to stderr."""
        if messages:
            fmt %= messages
        click.echo(fmt, file=sys.stderr)

    def vlog(self, fmt: str, *messages: Tuple[str]) -> None:
        """Log a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(fmt, *messages)


pass_context = click.make_pass_decorator(_Context, ensure=True)
cmd_folder = Path(__file__).parent.joinpath("commands").resolve()


class _ComplexCLI(click.MultiCommand):
    """Complex command-line options with subcommands for qaa."""

    def list_commands(self, ctx: click.Context) -> List[str]:
        """List available commands.

        Parameters
        ----------
        ctx : Context
            click context

        Returns
        -------
        List of str
            List of available commands
        """
        commands: List[str] = []
        for filename in cmd_folder.iterdir():
            if filename.suffix == ".py" and filename.stem.startswith("cmd_"):
                commands.append(filename.stem[4:])
        commands.sort()
        return commands

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[core.Command]:
        """Run the selected command.

        Parameters
        ----------
        ctx : Context
            click context
        cmd_name : str
            command name

        Returns
        -------
            The chosen command if present
        """
        try:
            if sys.version_info[0] == 2:
                cmd_name = cmd_name.encode("ascii", "replace")
            mod = __import__("qaa.commands.cmd_" + cmd_name, None, None, ["cli"])
        except ImportError:
            return None
        return mod.cli


@click.command(cls=_ComplexCLI, context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def main() -> None:
    """Run main command-line interface."""
    pass
