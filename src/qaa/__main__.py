"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Quasi-Anharmonic Analysis."""


if __name__ == "__main__":
    main(prog_name="qaa")  # pragma: no cover
