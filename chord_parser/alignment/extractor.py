"""Chord extraction from tab sheets.

This module provides functions to extract chord sequences from parsed
tab sheets for alignment with timed chord annotations.
"""

from __future__ import annotations

from chord_parser.alignment.models import TabChord
from chord_parser.tab_parser.models import Block, ChordLine, TabSheet


def extract_tab_chords(tab_sheet: TabSheet) -> tuple[TabChord, ...]:
    """Extract all chords from a parsed tab sheet.

    Iterates through all sections and items in the tab sheet,
    extracting chord tokens and creating TabChord objects with
    section context and global position indices.

    Parameters
    ----------
    tab_sheet : TabSheet
        The parsed tab sheet.

    Returns
    -------
    tuple[TabChord, ...]
        Ordered sequence of tab chords with section and position info.

    Examples
    --------
    >>> from chord_parser.tab_parser import parse
    >>> from chord_parser.alignment.extractor import extract_tab_chords
    >>> sheet = parse("[Verse]\\nGm C F\\nSome lyrics")
    >>> chords = extract_tab_chords(sheet)
    >>> [c.label for c in chords]
    ['Gm', 'C', 'F']
    """
    tab_chords: list[TabChord] = []
    index = 0

    for section in tab_sheet.sections:
        section_name = section.name if section.name else "Unknown"

        for item in section.items:
            tokens = []

            if isinstance(item, Block):
                # Block has chord_tokens attribute
                tokens = item.chord_tokens
            elif isinstance(item, ChordLine):
                # ChordLine has tokens attribute
                tokens = item.tokens
            else:
                # Skip other item types (LyricLine, EmptyLine, CommentLine)
                continue

            for token in tokens:
                # Only include chord tokens with parsed chords
                if token.kind == "chord" and token.chord is not None:
                    tab_chords.append(
                        TabChord(
                            chord=token.chord,
                            label=token.text,
                            section=section_name,
                            index=index,
                        )
                    )
                    index += 1

    return tuple(tab_chords)
