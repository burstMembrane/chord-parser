"""Vocal anchor extraction for multi-modal chord alignment.

This module extracts timing anchors for chords by matching chord-word
attachments from tabs to word-aligned lyrics from vocal transcription.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chord_parser.models import Chord

from chord_parser.alignment.smams import AlignedWord, WordAlignedLyrics, _normalize_word
from chord_parser.tab_parser.models import Block, ChordAttachment, ChordLine, Section, TabSheet


@dataclass(frozen=True)
class VocalAnchor:
    """A chord timing anchor derived from vocal alignment.

    Parameters
    ----------
    chord : Chord
        The parsed chord.
    chord_label : str
        Original chord text from tab.
    word : str
        The word the chord is attached to in the tab.
    aligned_word : AlignedWord
        The matched word from vocal alignment with timing.
    section : str
        Section name from the tab.
    sequence_index : int
        Position in the flattened chord sequence.
    confidence : float
        Confidence in the word match (0.0 to 1.0).
    """

    chord: Chord
    chord_label: str
    word: str
    aligned_word: AlignedWord
    section: str
    sequence_index: int
    confidence: float

    @property
    def time(self) -> float:
        """Get the anchor time in seconds."""
        return self.aligned_word.start


@dataclass(frozen=True)
class VocalAnchorResult:
    """Result of vocal anchor extraction.

    Parameters
    ----------
    anchors : tuple[VocalAnchor, ...]
        All extracted vocal anchors in sequence order.
    unanchored_count : int
        Number of chords without vocal anchors (floating or unmatched).
    total_chords : int
        Total number of chords in the tab.
    """

    anchors: tuple[VocalAnchor, ...]
    unanchored_count: int
    total_chords: int

    @property
    def coverage(self) -> float:
        """Fraction of chords with vocal anchors."""
        if self.total_chords == 0:
            return 0.0
        return len(self.anchors) / self.total_chords


def _match_word_to_alignment(
    word: str,
    word_alignment: WordAlignedLyrics,
    after_time: float,
    within_window: float = 30.0,
) -> tuple[AlignedWord | None, float]:
    """Match a tab word to a word in the vocal alignment.

    Uses normalized matching and returns the first match after the given time.
    Prioritizes exact matches over partial/substring matches.

    Parameters
    ----------
    word : str
        The word from the tab to match.
    word_alignment : WordAlignedLyrics
        The word-aligned lyrics from SMAMS.
    after_time : float
        Only consider words starting after this time.
    within_window : float
        Maximum time window to search (avoids matching words much later).

    Returns
    -------
    tuple[AlignedWord | None, float]
        The matched word and confidence score, or (None, 0.0).
    """
    normalized_query = _normalize_word(word)
    if not normalized_query:
        return None, 0.0

    # Minimum length to consider for partial matching (avoid single char matches)
    min_partial_len = 3

    # First pass: look for exact match
    for aligned in word_alignment.words:
        if aligned.start < after_time:
            continue
        if aligned.start > after_time + within_window:
            break

        normalized_aligned = _normalize_word(aligned.word)
        if normalized_query == normalized_aligned:
            return aligned, 1.0

    # Second pass: look for partial/substring match (only if query is long enough)
    if len(normalized_query) >= min_partial_len:
        for aligned in word_alignment.words:
            if aligned.start < after_time:
                continue
            if aligned.start > after_time + within_window:
                break

            normalized_aligned = _normalize_word(aligned.word)
            # Only match if the aligned word is also reasonably long
            if len(normalized_aligned) >= min_partial_len:
                if normalized_query in normalized_aligned or normalized_aligned in normalized_query:
                    return aligned, 0.8

    return None, 0.0


def _extract_chords_from_section(
    section: Section,
) -> list[tuple[str, str | None, str]]:
    """Extract chord labels and attached words from a section.

    Parameters
    ----------
    section : Section
        The tab section to process.

    Returns
    -------
    list[tuple[str, str | None, str]]
        List of (chord_label, attached_word, strategy) tuples.
    """
    results: list[tuple[str, str | None, str]] = []

    for item in section.items:
        if isinstance(item, Block):
            for att in item.attachments.chord_to_word:
                word_text = att.word.text if att.word else None
                results.append((att.chord.text, word_text, att.strategy))

    return results


def extract_vocal_anchors(
    tab: TabSheet,
    word_alignment: WordAlignedLyrics,
    skip_sections: set[str] | None = None,
) -> VocalAnchorResult:
    """Extract vocal anchors from a tab sheet using word-aligned lyrics.

    For each chord with a word attachment, attempts to find the corresponding
    word in the vocal alignment to derive timing.

    Parameters
    ----------
    tab : TabSheet
        The parsed tab sheet with chord-word attachments.
    word_alignment : WordAlignedLyrics
        Word-aligned lyrics from SMAMS.
    skip_sections : set[str] | None
        Section names to skip (e.g., {"Intro", "Outro"} for instrumental sections).

    Returns
    -------
    VocalAnchorResult
        The extracted vocal anchors.
    """
    from chord_parser.tab_parser.chord_detector import parse_chord

    skip_sections = skip_sections or {"Intro", "Outro", "Instrumental", "Solo"}

    anchors: list[VocalAnchor] = []
    sequence_index = 0
    unanchored_count = 0
    last_anchor_time = 0.0

    for section in tab.sections:
        section_name = section.name
        is_skip_section = any(skip in section_name for skip in skip_sections)

        for item in section.items:
            # Handle ChordLine items (instrumental sections)
            if isinstance(item, ChordLine):
                for token in item.tokens:
                    if token.kind == "chord":
                        # Count but don't anchor (no lyrics)
                        sequence_index += 1
                        unanchored_count += 1

            elif isinstance(item, Block):
                for att in item.attachments.chord_to_word:
                    chord_label = att.chord.text
                    parsed_chord = parse_chord(chord_label)

                    if parsed_chord is None:
                        sequence_index += 1
                        unanchored_count += 1
                        continue

                    # Skip sections without lyrics (intro, outro)
                    if is_skip_section or att.word is None:
                        sequence_index += 1
                        unanchored_count += 1
                        continue

                    word_text = att.word.text

                    # Try to match the word to vocal alignment
                    matched, confidence = _match_word_to_alignment(
                        word_text,
                        word_alignment,
                        after_time=last_anchor_time,
                    )

                    if matched is not None:
                        anchor = VocalAnchor(
                            chord=parsed_chord,
                            chord_label=chord_label,
                            word=word_text,
                            aligned_word=matched,
                            section=section_name,
                            sequence_index=sequence_index,
                            confidence=confidence,
                        )
                        anchors.append(anchor)
                        last_anchor_time = matched.start
                    else:
                        unanchored_count += 1

                    sequence_index += 1

    return VocalAnchorResult(
        anchors=tuple(anchors),
        unanchored_count=unanchored_count,
        total_chords=sequence_index,
    )
