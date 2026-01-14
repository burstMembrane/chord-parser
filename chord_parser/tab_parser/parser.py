"""Main tab sheet parser orchestration.

This module provides the main parse() function that orchestrates the full
tab sheet parsing pipeline.
"""

from __future__ import annotations

import re

from chord_parser.tab_parser.attachment import attach_chords_to_words
from chord_parser.tab_parser.chord_detector import classify_line, classify_tokens
from chord_parser.tab_parser.models import (
    Block,
    ChordLine,
    CommentLine,
    EmptyLine,
    Item,
    LyricLine,
    Section,
    TabSheet,
    Token,
)
from chord_parser.tab_parser.tokenizer import tokenize_line

# Section header pattern: [Section Name]
SECTION_HEADER_RE = re.compile(r"^\s*\[(.+?)\]\s*$")

# Default section name when no header is present
DEFAULT_SECTION = "Song"


def preprocess(text: str) -> list[str]:
    """Preprocess input text into lines.

    Normalizes line endings and preserves original line content
    (only strips the newline character).

    Parameters
    ----------
    text : str
        The raw input text.

    Returns
    -------
    list[str]
        List of lines without trailing newlines.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into lines, preserving content (only strip newline)
    return text.split("\n")


def extract_section_name(line: str) -> str | None:
    """Extract section name from a header line.

    Parameters
    ----------
    line : str
        The line to check.

    Returns
    -------
    str | None
        The section name if this is a header, None otherwise.

    Examples
    --------
    >>> extract_section_name("[Verse 1]")
    'Verse 1'
    >>> extract_section_name("Hello world")
    """
    match = SECTION_HEADER_RE.match(line)
    if match:
        return match.group(1)
    return None


def tokenize_and_classify(line: str) -> list[Token]:
    """Tokenize a line and classify all tokens.

    Parameters
    ----------
    line : str
        The line to process.

    Returns
    -------
    list[Token]
        Classified tokens.
    """
    tokens = tokenize_line(line)
    return classify_tokens(tokens)


def create_block(chord_line: str, lyric_line: str) -> Block:
    """Create a Block from chord and lyric line pair.

    Pads both lines to the same width and computes chord-to-word attachments.

    Parameters
    ----------
    chord_line : str
        The chord line (raw).
    lyric_line : str
        The lyric line (raw).

    Returns
    -------
    Block
        The aligned block with attachments.
    """
    # Compute width and pad both lines
    width = max(len(chord_line), len(lyric_line))
    chord_padded = chord_line.ljust(width)
    lyric_padded = lyric_line.ljust(width)

    # Tokenize and classify padded lines
    chord_tokens = tokenize_and_classify(chord_padded)
    lyric_tokens = tokenize_and_classify(lyric_padded)

    # Filter to chord tokens only for attachment
    chords_only = [t for t in chord_tokens if t.kind == "chord"]

    # Compute attachments
    attachments = attach_chords_to_words(chords_only, lyric_tokens)

    return Block(
        chord_raw=chord_padded,
        lyric_raw=lyric_padded,
        width=width,
        chord_tokens=tuple(chord_tokens),
        lyric_tokens=tuple(lyric_tokens),
        attachments=attachments,
    )


def parse_lines_to_items(lines: list[str]) -> list[Item]:
    """Parse lines into items, pairing chord and lyric lines into blocks.

    Parameters
    ----------
    lines : list[str]
        Lines to parse (no section headers).

    Returns
    -------
    list[Item]
        List of parsed items.
    """
    items: list[Item] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        tokens = tokenize_and_classify(line)
        line_type = classify_line(line, tokens)

        if line_type == "empty":
            items.append(EmptyLine())
            i += 1
            continue

        if line_type == "comment":
            items.append(CommentLine(raw=line))
            i += 1
            continue

        if line_type == "chord":
            # Check if next line is a lyric line (for pairing)
            if i + 1 < n:
                next_line = lines[i + 1]
                next_tokens = tokenize_and_classify(next_line)
                next_type = classify_line(next_line, next_tokens)

                if next_type == "lyric":
                    # Create a block
                    block = create_block(line, next_line)
                    items.append(block)
                    i += 2
                    continue

            # Standalone chord line
            items.append(ChordLine(raw=line, tokens=tuple(tokens)))
            i += 1
            continue

        if line_type == "lyric":
            # Standalone lyric line
            items.append(LyricLine(raw=line, tokens=tuple(tokens)))
            i += 1
            continue

        # Shouldn't reach here, but handle gracefully
        items.append(LyricLine(raw=line, tokens=tuple(tokens)))
        i += 1

    return items


def parse(text: str) -> TabSheet:
    """Parse a tab sheet into structured data.

    This is the main entry point for tab sheet parsing.

    Parameters
    ----------
    text : str
        The raw tab sheet text.

    Returns
    -------
    TabSheet
        Structured representation of the tab sheet.

    Examples
    --------
    >>> text = '''[Verse]
    ... Gm     C
    ... Hello  world
    ... '''
    >>> sheet = parse(text)
    >>> len(sheet.sections)
    1
    >>> sheet.sections[0].name
    'Verse'
    """
    lines = preprocess(text)

    # Group lines by section
    sections: list[Section] = []
    current_name = DEFAULT_SECTION
    current_lines: list[str] = []

    for line in lines:
        section_name = extract_section_name(line)
        if section_name is not None:
            # Save previous section if it has content
            if current_lines:
                items = parse_lines_to_items(current_lines)
                sections.append(Section(name=current_name, items=tuple(items)))

            # Start new section
            current_name = section_name
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        items = parse_lines_to_items(current_lines)
        sections.append(Section(name=current_name, items=tuple(items)))

    # If no sections were created, create an empty default section
    if not sections:
        sections.append(Section(name=DEFAULT_SECTION, items=()))

    return TabSheet(sections=tuple(sections), raw=text)
