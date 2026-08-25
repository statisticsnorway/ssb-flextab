"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB flextab."""


if __name__ == "__main__":
    main(prog_name="ssb-flextab")  # pragma: no cover
