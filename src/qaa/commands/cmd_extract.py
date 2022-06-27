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

import click
import numpy as np

from .. import _MASK, create_logging_dict
from ..libs import configparser, trajectory


@click.command("extract", short_help="Extract frames from a trajectory")
@click.option(
    "-c",
    "--config",
    "configfile",
    metavar="FILE",
    default=[
        Path.cwd().joinpath("config.yaml").as_posix(),
    ],
    show_default=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    help="Config file",
)
@click.option(
    "-i",
    "--input",
    "input",
    metavar="FILE",
    default=[
        Path.cwd().joinpath("positions.npy").as_posix(),
    ],
    show_default=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, resolve_path=True),
    help="Pre-existing data file",
)
@click.option(
    "-m",
    "--mask",
    default="ca",
    show_default=True,
    type=click.Choice(_MASK.keys()),
    help="Atom selection",
)
def cli(
    config: str,
    mask: str,
) -> None:
    """Write extracted frames to a new file."""
    start_time: float = time.perf_counter()

    config_data = configparser.ConfigParser(config)
    config_data.load()
    config_data.parse()

    # Create subdirectories
    Path(config_data.saveDir).mkdir(exist_ok=True)
    Path(config_data.figDir).mkdir(exist_ok=True)
    Path(config_data.logfile).parent.mkdir(exist_ok=True)

    # Setup logging
    level = logging.WARNING
    if config_data.verbose:
        level = logging.INFO
    elif config_data.debug:
        level = logging.DEBUG

    logging.config.dictConfig(create_logging_dict(config_data.logfile, level))
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info(f"Using Configuration File: {config}")

    # Extract aligned coordinates or calculate dihedrals.
    traj = trajectory.Trajectory(
        topology=config_data.topology,
        trajectory=config_data.trajfiles,
        skip=config_data.stepVal,
        mask=_MASK[mask],
        start_res=config_data.startRes,
        end_res=config_data.endRes,
    )

    analysis = config_data.analysis
    data = traj.get_positions() if analysis == "coordinates" else traj.get_dihedrals()

    # Save data to binary file.
    outfile = Path(config_data.saveDir) / Path(analysis).with_stem(".npy")
    with open(outfile, "wb") as out:
        logging.info(f"Saving {analysis} to {outfile}")
        np.save(out, data)

    # Calculate total execution time
    stop_time: float = time.perf_counter()
    dt: float = stop_time - start_time
    struct_time: time.struct_time = time.gmtime(dt)
    output: str = time.strftime("%H:%M:%S", struct_time)
    logger.info(f"Total execution time: {output}")
