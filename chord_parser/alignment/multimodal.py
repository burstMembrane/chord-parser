"""Multi-modal chord alignment using vocal anchors and audio detection.

This module implements a two-pass alignment strategy:
1. First pass: Use vocal anchors (tab chords -> lyrics -> timing)
2. Second pass: Use Chordino audio detection to fill instrumental parts

This combines the precision of vocal alignment with the coverage of
audio-based chord detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chord_parser.models import Chord

from chord_parser.alignment.aligner import chord_distance_pitchclass
from chord_parser.alignment.models import TabChord, TimedChord
from chord_parser.alignment.smams import WordAlignedLyrics
from chord_parser.alignment.vocal_anchors import VocalAnchor, VocalAnchorResult, extract_vocal_anchors
from chord_parser.tab_parser.models import Block, ChordLine, Section, TabSheet


@dataclass(frozen=True)
class MultimodalChord:
    """A chord with timing from multi-modal alignment.

    Parameters
    ----------
    chord : Chord
        The parsed chord.
    label : str
        Original chord label from tab.
    section : str
        Section name from tab.
    index : int
        Position in the flattened chord sequence.
    start : float
        Start time in seconds.
    end : float | None
        End time in seconds.
    source : str
        Timing source: "vocal" or "chordino".
    confidence : float
        Confidence in the timing (0.0 to 1.0).
    matched_audio_chord : TimedChord | None
        If source is "chordino", the matched audio chord.
    vocal_anchor : VocalAnchor | None
        If source is "vocal", the vocal anchor used.
    """

    chord: Chord
    label: str
    section: str
    index: int
    start: float
    end: float | None
    source: str
    confidence: float
    matched_audio_chord: TimedChord | None = None
    vocal_anchor: VocalAnchor | None = None


@dataclass(frozen=True)
class MultimodalAlignmentResult:
    """Result of multi-modal chord alignment.

    Parameters
    ----------
    chords : tuple[MultimodalChord, ...]
        All aligned chords in sequence order.
    vocal_anchors : VocalAnchorResult
        The vocal anchor extraction result.
    vocal_count : int
        Number of chords timed via vocal anchors.
    chordino_count : int
        Number of chords timed via Chordino.
    unmatched_count : int
        Number of chords without timing.
    """

    chords: tuple[MultimodalChord, ...]
    vocal_anchors: VocalAnchorResult
    vocal_count: int
    chordino_count: int
    unmatched_count: int

    @property
    def total_count(self) -> int:
        """Total number of chords."""
        return len(self.chords)

    @property
    def coverage(self) -> float:
        """Fraction of chords with timing."""
        if not self.chords:
            return 0.0
        return (self.vocal_count + self.chordino_count) / len(self.chords)


def _find_best_chordino_match(
    chord: Chord,
    audio_chords: list[TimedChord],
    after_time: float,
    before_time: float | None = None,
    distance_threshold: float = 0.5,
) -> tuple[TimedChord | None, float]:
    """Find the best matching Chordino chord in a time window.

    Parameters
    ----------
    chord : Chord
        The chord to match.
    audio_chords : list[TimedChord]
        Chordino chord detections.
    after_time : float
        Start of search window.
    before_time : float | None
        End of search window (None for no limit).
    distance_threshold : float
        Maximum chord distance to accept.

    Returns
    -------
    tuple[TimedChord | None, float]
        Best matching chord and distance, or (None, inf).
    """
    best_match: TimedChord | None = None
    best_distance = float("inf")

    for tc in audio_chords:
        if tc.start < after_time:
            continue
        if before_time is not None and tc.start > before_time:
            break
        if tc.chord is None:
            continue

        dist = chord_distance_pitchclass(chord, tc.chord)
        if dist < best_distance and dist <= distance_threshold:
            best_distance = dist
            best_match = tc

    return best_match, best_distance


def _extract_tab_chord_sequence(tab: TabSheet) -> list[tuple[Chord, str, str, int]]:
    """Extract flattened chord sequence from tab with metadata.

    Parameters
    ----------
    tab : TabSheet
        The parsed tab sheet.

    Returns
    -------
    list[tuple[Chord, str, str, int]]
        List of (chord, label, section_name, index) tuples.
    """
    from chord_parser.tab_parser.chord_detector import parse_chord

    sequence: list[tuple[Chord, str, str, int]] = []
    index = 0

    for section in tab.sections:
        for item in section.items:
            if isinstance(item, Block):
                for att in item.attachments.chord_to_word:
                    parsed = parse_chord(att.chord.text)
                    if parsed is not None:
                        sequence.append((parsed, att.chord.text, section.name, index))
                    index += 1
            elif isinstance(item, ChordLine):
                for token in item.tokens:
                    if token.kind == "chord":
                        parsed = parse_chord(token.text)
                        if parsed is not None:
                            sequence.append((parsed, token.text, section.name, index))
                        index += 1

    return sequence


def align_multimodal(
    tab: TabSheet,
    word_alignment: WordAlignedLyrics,
    audio_chords: list[TimedChord],
    instrumental_sections: set[str] | None = None,
    distance_threshold: float = 0.5,
) -> MultimodalAlignmentResult:
    """Perform multi-modal chord alignment.

    Two-pass alignment:
    1. First pass: Use vocal anchors for chords with lyrics
    2. Second pass: Use Chordino for instrumental sections

    Parameters
    ----------
    tab : TabSheet
        The parsed tab sheet.
    word_alignment : WordAlignedLyrics
        Word-aligned lyrics from SMAMS.
    audio_chords : list[TimedChord]
        Chordino chord detections.
    instrumental_sections : set[str] | None
        Section names that are instrumental (default: Intro, Outro, etc).
    distance_threshold : float
        Maximum chord distance for Chordino matching.

    Returns
    -------
    MultimodalAlignmentResult
        The multi-modal alignment result.
    """
    instrumental_sections = instrumental_sections or {"Intro", "Outro", "Instrumental", "Solo", "Interlude"}

    # Pass 1: Extract vocal anchors
    vocal_result = extract_vocal_anchors(tab, word_alignment, skip_sections=instrumental_sections)

    # Build index of vocal anchors by sequence index
    vocal_by_index: dict[int, VocalAnchor] = {a.sequence_index: a for a in vocal_result.anchors}

    # Extract full chord sequence from tab
    chord_sequence = _extract_tab_chord_sequence(tab)

    # Pass 2: Fill in with Chordino
    aligned_chords: list[MultimodalChord] = []
    vocal_count = 0
    chordino_count = 0
    unmatched_count = 0

    # Track time progression for Chordino matching
    last_known_time = 0.0

    for chord, label, section, idx in chord_sequence:
        is_instrumental = any(inst in section for inst in instrumental_sections)

        # Check if we have a vocal anchor for this chord
        if idx in vocal_by_index:
            anchor = vocal_by_index[idx]

            # Calculate end time from next anchor or Chordino
            end_time = None
            next_vocal = next(
                (a for a in vocal_result.anchors if a.sequence_index > idx),
                None,
            )
            if next_vocal:
                end_time = next_vocal.time

            aligned = MultimodalChord(
                chord=chord,
                label=label,
                section=section,
                index=idx,
                start=anchor.time,
                end=end_time,
                source="vocal",
                confidence=anchor.confidence,
                vocal_anchor=anchor,
            )
            aligned_chords.append(aligned)
            vocal_count += 1
            last_known_time = anchor.time

        elif is_instrumental:
            # Use Chordino for instrumental sections
            # Estimate time window based on last known position
            window_end = last_known_time + 30.0  # Search up to 30s ahead

            match, distance = _find_best_chordino_match(
                chord,
                audio_chords,
                after_time=last_known_time,
                before_time=window_end,
                distance_threshold=distance_threshold,
            )

            if match is not None:
                confidence = max(0.0, 1.0 - distance)
                aligned = MultimodalChord(
                    chord=chord,
                    label=label,
                    section=section,
                    index=idx,
                    start=match.start,
                    end=match.end,
                    source="chordino",
                    confidence=confidence,
                    matched_audio_chord=match,
                )
                aligned_chords.append(aligned)
                chordino_count += 1
                last_known_time = match.start
            else:
                # No match found
                aligned = MultimodalChord(
                    chord=chord,
                    label=label,
                    section=section,
                    index=idx,
                    start=last_known_time,
                    end=None,
                    source="unmatched",
                    confidence=0.0,
                )
                aligned_chords.append(aligned)
                unmatched_count += 1
        else:
            # Non-instrumental section but no vocal anchor
            # This shouldn't happen often - try Chordino as fallback
            window_end = last_known_time + 30.0

            match, distance = _find_best_chordino_match(
                chord,
                audio_chords,
                after_time=last_known_time,
                before_time=window_end,
                distance_threshold=distance_threshold,
            )

            if match is not None:
                confidence = max(0.0, 1.0 - distance)
                aligned = MultimodalChord(
                    chord=chord,
                    label=label,
                    section=section,
                    index=idx,
                    start=match.start,
                    end=match.end,
                    source="chordino",
                    confidence=confidence,
                    matched_audio_chord=match,
                )
                aligned_chords.append(aligned)
                chordino_count += 1
                last_known_time = match.start
            else:
                aligned = MultimodalChord(
                    chord=chord,
                    label=label,
                    section=section,
                    index=idx,
                    start=last_known_time,
                    end=None,
                    source="unmatched",
                    confidence=0.0,
                )
                aligned_chords.append(aligned)
                unmatched_count += 1

    return MultimodalAlignmentResult(
        chords=tuple(aligned_chords),
        vocal_anchors=vocal_result,
        vocal_count=vocal_count,
        chordino_count=chordino_count,
        unmatched_count=unmatched_count,
    )
