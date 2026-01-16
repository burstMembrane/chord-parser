"""Tab sheet parser for chord/lyric alignment.

This module provides functionality to parse plain-text chord sheets into
structured data with section detection, chord-to-word alignment, and
bidirectional mappings.
"""

from chord_parser.tab_parser.models import (
    AttachmentResult,
    Block,
    ChordAttachment,
    ChordLine,
    CommentLine,
    EmptyLine,
    Item,
    LyricLine,
    Section,
    TabSheet,
    Token,
    WordAttachment,
)
from chord_parser.tab_parser.parser import parse, parse_chords, parse_flat

__all__ = [
    "AttachmentResult",
    "Block",
    "ChordAttachment",
    "ChordLine",
    "CommentLine",
    "EmptyLine",
    "Item",
    "LyricLine",
    "Section",
    "TabSheet",
    "Token",
    "WordAttachment",
    "parse",
    "parse_chords",
    "parse_flat",
]
