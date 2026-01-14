"""Data models for tab sheet parsing.

This module defines the core data structures for representing parsed
tab sheets, including tokens, blocks, sections, and attachment mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from chord_parser.models import Chord


TokenKind = Literal["chord", "word", "punct", "other"]


@dataclass(frozen=True)
class Token:
    """A token with column span information.

    Parameters
    ----------
    text : str
        The token text content.
    start : int
        Inclusive start column (0-indexed).
    end : int
        Exclusive end column.
    kind : TokenKind
        The token classification.
    chord : Chord | None
        Parsed Chord object if kind is "chord", None otherwise.

    Examples
    --------
    >>> token = Token(text="Gm7", start=0, end=3, kind="chord")
    >>> token.start, token.end
    (0, 3)
    """

    text: str
    start: int
    end: int
    kind: TokenKind
    chord: Chord | None = None


@dataclass(frozen=True)
class ChordAttachment:
    """Links a chord token to its attached word.

    Parameters
    ----------
    chord : Token
        The chord token.
    word : Token | None
        The word token this chord attaches to, or None if floating.
    strategy : str
        The attachment strategy used (e.g., "overlap", "nearest_left").
    """

    chord: Token
    word: Token | None
    strategy: str


@dataclass(frozen=True)
class WordAttachment:
    """Links a word token to its attached chords.

    Parameters
    ----------
    word : Token
        The word token.
    chords : tuple[Token, ...]
        Chord tokens attached to this word, ordered by position.
    """

    word: Token
    chords: tuple[Token, ...]


@dataclass(frozen=True)
class AttachmentResult:
    """Bidirectional chord-word attachments.

    Parameters
    ----------
    chord_to_word : tuple[ChordAttachment, ...]
        Mapping from each chord to its attached word.
    word_to_chords : tuple[WordAttachment, ...]
        Mapping from each word to its attached chords.
    """

    chord_to_word: tuple[ChordAttachment, ...]
    word_to_chords: tuple[WordAttachment, ...]


@dataclass(frozen=True)
class Block:
    """An aligned chord+lyric line pair.

    Parameters
    ----------
    chord_raw : str
        The raw chord line (right-padded to width).
    lyric_raw : str
        The raw lyric line (right-padded to width).
    width : int
        The padded width of both lines.
    chord_tokens : tuple[Token, ...]
        Tokenized chord line.
    lyric_tokens : tuple[Token, ...]
        Tokenized lyric line.
    attachments : AttachmentResult
        Chord-to-word and word-to-chord mappings.
    """

    chord_raw: str
    lyric_raw: str
    width: int
    chord_tokens: tuple[Token, ...]
    lyric_tokens: tuple[Token, ...]
    attachments: AttachmentResult


@dataclass(frozen=True)
class ChordLine:
    """A line containing only chords (e.g., intro/outro).

    Parameters
    ----------
    raw : str
        The raw line text.
    tokens : tuple[Token, ...]
        Tokenized content.
    """

    raw: str
    tokens: tuple[Token, ...]


@dataclass(frozen=True)
class LyricLine:
    """A line containing lyrics (possibly with no chords above).

    Parameters
    ----------
    raw : str
        The raw line text.
    tokens : tuple[Token, ...]
        Tokenized content.
    """

    raw: str
    tokens: tuple[Token, ...]


@dataclass(frozen=True)
class EmptyLine:
    """An empty or whitespace-only line."""

    pass


@dataclass(frozen=True)
class CommentLine:
    """A comment line (e.g., "(loop and fade)").

    Parameters
    ----------
    raw : str
        The raw line text.
    """

    raw: str


Item = Block | ChordLine | LyricLine | EmptyLine | CommentLine


@dataclass(frozen=True)
class Section:
    """A named section containing items.

    Parameters
    ----------
    name : str
        The section name (e.g., "Verse 1", "Chorus").
    items : tuple[Item, ...]
        The items within this section.
    """

    name: str
    items: tuple[Item, ...]


@dataclass(frozen=True)
class TabSheet:
    """Complete parsed tab sheet.

    Parameters
    ----------
    sections : tuple[Section, ...]
        All sections in the tab sheet.
    raw : str
        The original input text.
    """

    sections: tuple[Section, ...]
    raw: str
