"""Tools for creating flexible summary tables with pandas."""

from .flextab import flextab
from .formatting import flextab_to_markdown
from .formatting import flextab_to_string
from .result import FlextabResult

__all__ = [
    "FlextabResult",
    "flextab",
    "flextab_to_markdown",
    "flextab_to_string",
]
