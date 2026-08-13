"""Shared Rich theme for every console the CLI creates.

Plain ``bold``/``italic`` headers are a trap on light terminals: many themes
render bold text with the bright-black palette color, which reads as washed-out
gray on a white background. Pinning table headers and titles to a deep blue
keeps them readable on both light and dark terminals.
"""

from __future__ import annotations

from rich.theme import Theme

THEME = Theme(
    {
        "table.header": "bold blue3",
        "table.title": "bold blue3",
    }
)
