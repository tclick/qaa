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
"""Extract frames from a trajectory into a new trajectory.

A trajectory will be read, and the selected frames will be extracted into a new file for
further processing.
"""
import logging.config
import time
from pathlib import Path
from typing import Any

import click

from .. import _MASK, create_logging_dict
from ..libs import configparser, trajectory


@click.command("extract", short_help="Extract frames from a trajectory")
@click.option(
    "-c",
    "--config",
    metavar="FILE",
    is_eager=True,
    expose_value=False,
    default=Path.cwd().joinpath("config.yaml"),
    show_default=True,
    callback=configparser.configure,
    type=click.Path(file_okay=True, resolve_path=True, path_type=Path),
    help="Read option defaults from the specified YAML file",
)
@click.option(
    "-a",
    "--analysis",
    default="coordinates",
    show_default=True,
    type=click.Choice(["coordinates", "dihedrals"]),
    help="Analysis type",
)
@click.option("--verbose/--no-verbose", default=False, help="Verbose output")
@click.option("--debug/--no-debug", default=False, help="Debugging output")
@click.option(
    "-p",
    "--topology",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    default="protein.parm7",
    help="Topology file",
)
@click.option(
    "-f",
    "--trajfiles",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
    help="List of trajectories files",
)
@click.option(
    "--trajform",
    type=(
        click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
        str,
    ),
    help="Trajectory form and included frames [ex: protein-***.nc 1:10]",
)
@click.option(
    "-b",
    "--startres",
    default=1,
    type=click.IntRange(min=1),
    show_default=True,
    help="Starting residue for analysis",
)
@click.option(
    "-e",
    "--endres",
    default=10,
    type=click.INT,
    show_default=True,
    help="Final residue for analysis",
)
@click.option(
    "--skip",
    default=1,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of frames to skip",
)
@click.option(
    "-o",
    "--outdir",
    metavar="DIR",
    default=Path.cwd() / "savefiles",
    show_default=True,
    type=click.Path(file_okay=False, resolve_path=True, path_type=Path),
    help="Output directory",
)
@click.option(
    "-l",
    "--logfile",
    metavar="FILE",
    default=Path.cwd() / "log" / "log.txt",
    show_default=True,
    type=click.Path(dir_okay=False, resolve_path=True, path_type=Path),
    help="Log file",
)
@click.option(
    "-m",
    "--mask",
    metavar="MASK",
    default="ca",
    show_default=True,
    type=click.Choice(_MASK.keys()),
    help="Atom selection",
)
@click.option(
    "--align / --no-align", default=True, help="Align coordinate trajectories"
)
def cli(**kwargs: Any) -> None:
    """Write extracted frames to a new file."""
    start_time: float = time.perf_counter()

    # Create subdirectories
    parser = configparser.Config()
    parser.update(**kwargs)
    configparser.parse(parser)

    Path(parser.outdir).mkdir(exist_ok=True)
    Path(parser.logfile).parent.mkdir(exist_ok=True)

    # Setup logging
    level = logging.WARNING
    if parser.verbose:
        level = logging.INFO
    elif parser.debug:
        level = logging.DEBUG

    logging.config.dictConfig(create_logging_dict(parser.logfile, level))
    logger: logging.Logger = logging.getLogger(__name__)
    if parser.debug:
        for k, v in kwargs.items():
            logger.debug(f"{k}: {v}")

    # Extract aligned coordinates or calculate dihedrals.
    traj = trajectory.Trajectory(
        parser.topology,
        *parser.trajfiles,
        skip=parser.skip,
        mask=_MASK[parser.mask],
        start_res=parser.startres,
        end_res=parser.endres,
    )

    analysis = parser.analysis
    filename = parser.outdir / Path(parser.analysis).with_suffix(".npy")
    if analysis == "coordinates":
        traj.get_positions(filename, align=parser.align)
    else:
        traj.get_dihedrals(filename)

    # Calculate total execution time
    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    output: str = time.strftime("%H:%M:%S", struct_time)
    logger.info(f"Total execution time: {output}")
