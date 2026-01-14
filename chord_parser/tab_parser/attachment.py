"""Chord-to-word attachment logic for tab sheets.

This module provides functions to attach chord tokens to their corresponding
lyric tokens based on column overlap and fallback strategies.
"""

from __future__ import annotations

from chord_parser.tab_parser.models import (
    AttachmentResult,
    ChordAttachment,
    Token,
    WordAttachment,
)


def compute_overlap(chord: Token, word: Token) -> int:
    """Compute the column overlap between a chord and word token.

    Parameters
    ----------
    chord : Token
        The chord token.
    word : Token
        The word token.

    Returns
    -------
    int
        The number of overlapping columns (0 if no overlap).

    Examples
    --------
    >>> from chord_parser.tab_parser.models import Token
    >>> c = Token(text="Gm", start=0, end=2, kind="chord")
    >>> w = Token(text="Hello", start=0, end=5, kind="word")
    >>> compute_overlap(c, w)
    2
    """
    overlap_start = max(chord.start, word.start)
    overlap_end = min(chord.end, word.end)
    return max(0, overlap_end - overlap_start)


def find_best_overlap(chord: Token, words: list[Token]) -> tuple[Token | None, int]:
    """Find the word with maximum overlap for a chord.

    Parameters
    ----------
    chord : Token
        The chord token.
    words : list[Token]
        List of word tokens to search.

    Returns
    -------
    tuple[Token | None, int]
        The best matching word and its overlap (None, 0 if no overlap).
    """
    best_word: Token | None = None
    best_overlap = 0

    for word in words:
        overlap = compute_overlap(chord, word)
        if overlap > best_overlap:
            best_overlap = overlap
            best_word = word

    return best_word, best_overlap


def find_nearest_left(chord: Token, words: list[Token]) -> Token | None:
    """Find the nearest word to the left of a chord.

    Parameters
    ----------
    chord : Token
        The chord token.
    words : list[Token]
        List of word tokens to search.

    Returns
    -------
    Token | None
        The nearest word with start <= chord.start, or None.
    """
    best_word: Token | None = None
    best_start = -1

    for word in words:
        if word.start <= chord.start and word.start > best_start:
            best_start = word.start
            best_word = word

    return best_word


def find_nearest_right(chord: Token, words: list[Token]) -> Token | None:
    """Find the nearest word to the right of a chord.

    Parameters
    ----------
    chord : Token
        The chord token.
    words : list[Token]
        List of word tokens to search.

    Returns
    -------
    Token | None
        The nearest word with start > chord.start, or None.
    """
    best_word: Token | None = None
    best_start = float("inf")

    for word in words:
        if word.start > chord.start and word.start < best_start:
            best_start = word.start
            best_word = word

    return best_word


def attach_chord_to_word(chord: Token, words: list[Token]) -> ChordAttachment:
    """Attach a single chord to the best matching word.

    Strategy:
    1. Find the word with maximum column overlap
    2. If no overlap, find the nearest word to the left
    3. If no left word, find the nearest word to the right
    4. If no words at all, chord floats (word=None)

    Parameters
    ----------
    chord : Token
        The chord token to attach.
    words : list[Token]
        List of word tokens (lyric tokens with kind="word").

    Returns
    -------
    ChordAttachment
        The attachment with strategy information.
    """
    if not words:
        return ChordAttachment(chord=chord, word=None, strategy="no_words")

    # Strategy 1: overlap
    best_word, overlap = find_best_overlap(chord, words)
    if overlap > 0 and best_word is not None:
        return ChordAttachment(chord=chord, word=best_word, strategy="overlap")

    # Strategy 2: nearest left
    left_word = find_nearest_left(chord, words)
    if left_word is not None:
        return ChordAttachment(chord=chord, word=left_word, strategy="nearest_left")

    # Strategy 3: nearest right
    right_word = find_nearest_right(chord, words)
    if right_word is not None:
        return ChordAttachment(chord=chord, word=right_word, strategy="nearest_right")

    # No attachment possible
    return ChordAttachment(chord=chord, word=None, strategy="floating")


def attach_chords_to_words(
    chord_tokens: list[Token],
    lyric_tokens: list[Token],
) -> AttachmentResult:
    """Compute bidirectional chord-to-word attachments.

    Parameters
    ----------
    chord_tokens : list[Token]
        List of chord tokens from the chord line.
    lyric_tokens : list[Token]
        List of all tokens from the lyric line.

    Returns
    -------
    AttachmentResult
        Bidirectional mappings between chords and words.

    Examples
    --------
    >>> from chord_parser.tab_parser.models import Token
    >>> chords = [Token("Gm", 0, 2, "chord"), Token("C", 7, 8, "chord")]
    >>> lyrics = [Token("Hello", 0, 5, "word"), Token("world", 7, 12, "word")]
    >>> result = attach_chords_to_words(chords, lyrics)
    >>> result.chord_to_word[0].word.text
    'Hello'
    """
    # Filter to only word tokens for attachment
    words = [t for t in lyric_tokens if t.kind == "word"]

    # Build chord -> word mapping
    chord_to_word: list[ChordAttachment] = []
    for chord in chord_tokens:
        attachment = attach_chord_to_word(chord, words)
        chord_to_word.append(attachment)

    # Build word -> chords mapping (inverse)
    word_to_chords_map: dict[Token, list[Token]] = {w: [] for w in words}

    for attachment in chord_to_word:
        if attachment.word is not None and attachment.word in word_to_chords_map:
            word_to_chords_map[attachment.word].append(attachment.chord)

    # Sort chords by position for each word
    word_to_chords: list[WordAttachment] = []
    for word in words:
        chords = sorted(word_to_chords_map[word], key=lambda c: c.start)
        word_to_chords.append(WordAttachment(word=word, chords=tuple(chords)))

    return AttachmentResult(
        chord_to_word=tuple(chord_to_word),
        word_to_chords=tuple(word_to_chords),
    )
