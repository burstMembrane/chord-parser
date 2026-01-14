"""Chord detection and line classification for tab sheets.

This module provides regex-based chord detection and functions to classify
tokens and lines in a tab sheet.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from chord_parser.converter import from_pychord
from chord_parser.tab_parser.models import Token

if TYPE_CHECKING:
    from chord_parser.models import Chord

# Constants for chord detection
MAX_CHORD_LENGTH = 15
CHORD_LINE_THRESHOLD = 0.6

# Regex pattern for chord detection
# Matches: root (A-G), optional accidental (b/#), optional quality, optional slash bass
CHORD_RE = re.compile(
    r"^[A-G][b#]?"  # Root note with optional accidental
    r"(?:"
    r"m(?:aj)?(?:7|9|11|13)?|"  # minor variants: m, maj, maj7, m7, m9, etc.
    r"M(?:aj)?(?:7|9|11|13)?|"  # major variants: M, Maj, Maj7, M7, etc.
    r"dim(?:7)?|"  # diminished
    r"aug(?:7)?|"  # augmented
    r"sus[24]?(?:7)?|"  # suspended
    r"add[29]|"  # added tones
    r"[679]|"  # extensions
    r"7|9|11|13|"  # dominant extensions
    r"m7-5|m7b5|"  # half-diminished
    r"mM7|mmaj7|"  # minor-major seventh
    r"5"  # power chord
    r")*"
    r"(?:/[A-G][b#]?)?$",  # Optional slash bass
    re.IGNORECASE,
)

# Known false positives - lowercase words that might match chord pattern
# Only reject these when they appear in lowercase (lyrics context)
# Uppercase versions like "Am", "G" are valid chords
FALSE_POSITIVES_LOWERCASE: set[str] = {
    "a",
    "am",
    "be",
}

LineType = Literal["chord", "lyric", "empty", "comment", "section_header"]


def is_chord(text: str) -> bool:
    """Check if text matches chord pattern.

    Uses regex pre-filtering followed by pychord validation.

    Parameters
    ----------
    text : str
        The text to check.

    Returns
    -------
    bool
        True if the text is a valid chord, False otherwise.

    Examples
    --------
    >>> is_chord("Gm7")
    True
    >>> is_chord("Hello")
    False
    >>> is_chord("C/E")
    True
    """
    # Quick rejection for obvious non-chords
    if not text or len(text) > MAX_CHORD_LENGTH:
        return False

    # Check false positives - only reject if text is all lowercase
    # This prevents "a" and "am" in lyrics from being detected as chords
    # but allows "A" and "Am" as valid chords
    if text.islower() and text in FALSE_POSITIVES_LOWERCASE:
        return False

    # Regex pre-filter
    if not CHORD_RE.match(text):
        return False

    # Validate with pychord
    return parse_chord(text) is not None


def parse_chord(text: str) -> Chord | None:
    """Parse a chord string into a Chord object.

    Parameters
    ----------
    text : str
        The chord string to parse.

    Returns
    -------
    Chord | None
        The parsed Chord object, or None if parsing fails.

    Examples
    --------
    >>> chord = parse_chord("Gm7")
    >>> chord is not None
    True
    >>> chord.quality
    'min7'
    """
    try:
        return from_pychord(text)
    except (ValueError, Exception):  # pychord may raise various exceptions
        return None


def classify_token(token: Token) -> Token:
    """Classify a single token as chord, word, punct, or other.

    Parameters
    ----------
    token : Token
        The token to classify (with kind="other").

    Returns
    -------
    Token
        A new token with updated kind and chord fields.

    Examples
    --------
    >>> from chord_parser.tab_parser.models import Token
    >>> t = Token(text="Gm7", start=0, end=3, kind="other")
    >>> classified = classify_token(t)
    >>> classified.kind
    'chord'
    """
    text = token.text

    # Check if it's a chord
    chord = parse_chord(text) if is_chord(text) else None
    if chord is not None:
        return Token(
            text=text,
            start=token.start,
            end=token.end,
            kind="chord",
            chord=chord,
        )

    # Check if it's punctuation only
    if all(not c.isalnum() for c in text):
        return Token(
            text=text,
            start=token.start,
            end=token.end,
            kind="punct",
        )

    # Check if it contains alphabetic characters (word)
    if any(c.isalpha() for c in text):
        return Token(
            text=text,
            start=token.start,
            end=token.end,
            kind="word",
        )

    # Default to other
    return token


def classify_tokens(tokens: list[Token]) -> list[Token]:
    """Classify all tokens in a list.

    Parameters
    ----------
    tokens : list[Token]
        List of tokens with kind="other".

    Returns
    -------
    list[Token]
        List of tokens with updated kind and chord fields.
    """
    return [classify_token(t) for t in tokens]


def classify_line(line: str, tokens: list[Token] | None = None) -> LineType:
    """Classify a line based on its content.

    Parameters
    ----------
    line : str
        The line to classify.
    tokens : list[Token] | None
        Pre-classified tokens, or None to classify internally.

    Returns
    -------
    LineType
        The line classification.

    Examples
    --------
    >>> classify_line("")
    'empty'
    >>> classify_line("[Verse 1]")
    'section_header'
    >>> classify_line("(repeat)")
    'comment'
    """
    # Empty line
    stripped = line.strip()
    if not stripped:
        return "empty"

    # Section header: [Something]
    if re.match(r"^\s*\[.+?\]\s*$", line):
        return "section_header"

    # Comment line: (something) - typically instructions
    if re.match(r"^\s*\(.+?\)\s*$", line):
        return "comment"

    # Classify tokens if not provided
    if tokens is None:
        from chord_parser.tab_parser.tokenizer import tokenize_line

        raw_tokens = tokenize_line(line)
        tokens = classify_tokens(raw_tokens)

    if not tokens:
        return "empty"

    # Count chord and non-chord tokens
    chord_count = sum(1 for t in tokens if t.kind == "chord")
    word_count = sum(1 for t in tokens if t.kind == "word")
    total = len(tokens)

    # Calculate chord ratio
    chord_ratio = chord_count / total if total > 0 else 0

    # Chord line: high ratio of chords, no words
    if chord_ratio >= CHORD_LINE_THRESHOLD and chord_count > 0 and word_count == 0:
        return "chord"

    # Default to lyric line
    return "lyric"
