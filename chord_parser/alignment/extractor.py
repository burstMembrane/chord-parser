"""Extract chord sequences from parsed tab sheets.

This module provides functionality to flatten a TabSheet's hierarchical
structure into a linear sequence of chords for alignment.
"""

from __future__ import annotations

from chord_parser.alignment.models import TabChord
from chord_parser.tab_parser.models import Block, ChordLine, Section, TabSheet


def extract_chords_from_section(section: Section, start_index: int) -> list[TabChord]:
    """Extract all chords from a section in order.

    Parameters
    ----------
    section : Section
        The section to extract chords from.
    start_index : int
        Starting index for chords in this section.

    Returns
    -------
    list[TabChord]
        List of chords from this section.
    """
    chords: list[TabChord] = []
    index = start_index

    for item in section.items:
        if isinstance(item, Block):
            for token in item.chord_tokens:
                if token.kind == "chord" and token.chord is not None:
                    chords.append(
                        TabChord(
                            chord=token.chord,
                            label=token.text,
                            section=section.name,
                            index=index,
                        )
                    )
                    index += 1
        elif isinstance(item, ChordLine):
            for token in item.tokens:
                if token.kind == "chord" and token.chord is not None:
                    chords.append(
                        TabChord(
                            chord=token.chord,
                            label=token.text,
                            section=section.name,
                            index=index,
                        )
                    )
                    index += 1

    return chords


def extract_tab_chords(sheet: TabSheet) -> list[TabChord]:
    r"""Extract a flat list of chords from a tab sheet.

    Processes all sections in order, extracting chords from both
    Block items (chord+lyric pairs) and standalone ChordLine items.

    Parameters
    ----------
    sheet : TabSheet
        The parsed tab sheet.

    Returns
    -------
    list[TabChord]
        All chords in order of appearance.

    Examples
    --------
    >>> from chord_parser import tab_parser
    >>> sheet = tab_parser.parse("[Intro]\nGm C F\n[Verse]\nGm C\nHello")
    >>> chords = extract_tab_chords(sheet)
    >>> len(chords)
    5
    """
    all_chords: list[TabChord] = []
    index = 0

    for section in sheet.sections:
        section_chords = extract_chords_from_section(section, index)
        all_chords.extend(section_chords)
        index += len(section_chords)

    return all_chords
