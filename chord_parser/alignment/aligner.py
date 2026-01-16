"""Greedy forward chord alignment.

This module implements a simple greedy forward matcher for aligning
tab chords with timed audio chord annotations. Much simpler than DTW
and works better for this use case since both sequences are ordered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chord_parser.models import Chord

from chord_parser.alignment.models import (
    AlignedChord,
    AlignmentResult,
    TabChord,
    TimedChord,
)
from chord_parser.pitch_class import (
    roots_match,
    chord_pitch_similarity,
    quality_category,
    transpose_chord,
)

# Type alias for distance functions
DistanceFn = Callable[["Chord | None", "Chord | None"], float]


def chord_similarity(chord1: Chord | None, chord2: Chord | None) -> float:
    """Compute similarity between two chords.

    Uses a simple tiered approach:
    - 1.0: Exact match (root, quality, bass all identical)
    - 0.8: Same root, different quality (e.g., Am vs Am7)
    - 0.5: Bass note matches root (e.g., C/G vs G)
    - 0.0: No match

    Parameters
    ----------
    chord1 : Chord | None
        First chord (None represents "no chord").
    chord2 : Chord | None
        Second chord (None represents "no chord").

    Returns
    -------
    float
        Similarity between 0.0 (no match) and 1.0 (identical).
    """
    if chord1 is None or chord2 is None:
        return 0.0

    # Exact match (using enharmonic equivalence for roots)
    if roots_match(chord1, chord2) and chord1.quality == chord2.quality and chord1.bass == chord2.bass:
        return 1.0

    # Same root, different quality (e.g., Am vs Am7)
    if roots_match(chord1, chord2):
        return 0.8

    # Bass note of slash chord matches root
    if (chord1.bass is not None and chord1.bass == chord2.root) or (
        chord2.bass is not None and chord2.bass == chord1.root
    ):
        return 0.5

    return 0.0


def chord_similarity_weighted(
    chord1: Chord | None,
    chord2: Chord | None,
    *,
    weight_root: float = 0.5,
    weight_pitch: float = 0.3,
    weight_quality: float = 0.2,
    root_gate: bool = True,
) -> float:
    """Compute weighted similarity between two chords.

    Combines root match, pitch class Jaccard, and quality category match.
    This follows DECIBEL's approach of comparing chords by their actual
    note content rather than just string labels.

    Parameters
    ----------
    chord1 : Chord | None
        First chord.
    chord2 : Chord | None
        Second chord.
    weight_root : float
        Weight for root match component (default 0.5).
    weight_pitch : float
        Weight for pitch class Jaccard (default 0.3).
    weight_quality : float
        Weight for quality category match (default 0.2).
    root_gate : bool
        If True, return 0.0 when roots don't match (default True).

    Returns
    -------
    float
        Weighted similarity (0.0 to 1.0).
    """
    if chord1 is None or chord2 is None:
        return 0.0

    # Root match (enharmonic equivalence)
    root_match = 1.0 if roots_match(chord1, chord2) else 0.0

    # If root gate is enabled and roots don't match, return 0
    if root_gate and root_match == 0.0:
        return 0.0

    # Pitch class Jaccard similarity
    pitch_sim = chord_pitch_similarity(chord1, chord2)

    # Quality category match
    cat1 = quality_category(chord1.quality)
    cat2 = quality_category(chord2.quality)
    quality_match = 1.0 if cat1 == cat2 else 0.0

    # Weighted combination
    total_weight = weight_root + weight_pitch + weight_quality
    if total_weight == 0:
        return 0.0

    return (
        weight_root * root_match + weight_pitch * pitch_sim + weight_quality * quality_match
    ) / total_weight


def chord_distance_flexible(chord1: Chord | None, chord2: Chord | None) -> float:
    """Compute distance between two chords (1 - similarity).

    Kept for backward compatibility with existing code.

    Parameters
    ----------
    chord1 : Chord | None
        First chord.
    chord2 : Chord | None
        Second chord.

    Returns
    -------
    float
        Distance between 0.0 (identical) and 1.0 (no match).
    """
    # Special case: both None means "no chord" matches "no chord"
    if chord1 is None and chord2 is None:
        return 0.0
    return 1.0 - chord_similarity(chord1, chord2)


def chord_distance_exact(chord1: Chord | None, chord2: Chord | None) -> float:
    """Compute exact match distance between two chords.

    Parameters
    ----------
    chord1 : Chord | None
        First chord (None represents "no chord").
    chord2 : Chord | None
        Second chord (None represents "no chord").

    Returns
    -------
    float
        0.0 if chords are identical, 1.0 otherwise.
    """
    if chord1 is None and chord2 is None:
        return 0.0
    if chord1 is None or chord2 is None:
        return 1.0

    if chord1.root == chord2.root and chord1.quality == chord2.quality and chord1.bass == chord2.bass:
        return 0.0

    return 1.0


def align_chords(
    tab_chords: list[TabChord],
    timed_chords: list[TimedChord],
    lookahead: int = 5,
    min_similarity: float = 0.5,
    **_kwargs: object,
) -> AlignmentResult:
    """Align tab chords to timed chords using greedy forward matching.

    For each tab chord, searches within a lookahead window in the timed
    sequence to find the best match. Maintains monotonic order - once a
    timed chord is matched, only later timed chords are considered.

    Parameters
    ----------
    tab_chords : list[TabChord]
        Chords extracted from a tab sheet.
    timed_chords : list[TimedChord]
        Timed chord annotations from audio analysis.
    lookahead : int
        Number of timed chords to search ahead for each tab chord.
        Default is 5.
    min_similarity : float
        Minimum similarity score to accept a match (0.0 to 1.0).
        Default is 0.5.
    **_kwargs
        Ignored. Accepts extra kwargs for backward compatibility with
        old DTW-based API (step_pattern, distance_fn, open_begin, open_end).

    Returns
    -------
    AlignmentResult
        The alignment result with matched chord pairs.
    """
    # Filter out "N" (no chord) from timed chords
    filtered_timed = [tc for tc in timed_chords if tc.chord is not None]

    if not tab_chords or not filtered_timed:
        return AlignmentResult(
            alignments=(),
            tab_chords=tuple(tab_chords),
            timed_chords=tuple(timed_chords),
            total_distance=float("inf"),
            normalized_distance=float("inf"),
        )

    aligned: list[AlignedChord] = []
    total_distance = 0.0
    timed_idx = 0  # Current position in timed sequence

    for tab_chord in tab_chords:
        best_match_idx: int | None = None
        best_similarity = 0.0

        # Search within lookahead window
        search_end = min(timed_idx + lookahead, len(filtered_timed))
        for j in range(timed_idx, search_end):
            timed_chord = filtered_timed[j]
            sim = chord_similarity(tab_chord.chord, timed_chord.chord)

            if sim > best_similarity:
                best_similarity = sim
                best_match_idx = j

        # Accept match if above threshold
        if best_match_idx is not None and best_similarity >= min_similarity:
            matched_timed = filtered_timed[best_match_idx]
            distance = 1.0 - best_similarity

            aligned.append(
                AlignedChord(
                    tab_chord=tab_chord,
                    timed_chord=matched_timed,
                    distance=distance,
                )
            )
            total_distance += distance
            # Advance past the matched chord to maintain monotonic order
            timed_idx = best_match_idx + 1
        else:
            # No match found - skip "N" chords and try again
            while timed_idx < len(filtered_timed) and filtered_timed[timed_idx].label == "N":
                timed_idx += 1

    # Compute normalized distance
    normalized_distance = total_distance / len(aligned) if aligned else float("inf")

    return AlignmentResult(
        alignments=tuple(aligned),
        tab_chords=tuple(tab_chords),
        timed_chords=tuple(timed_chords),
        total_distance=total_distance,
        normalized_distance=normalized_distance,
    )


def align_with_transposition(
    tab_chords: list[TabChord],
    timed_chords: list[TimedChord],
    lookahead: int = 5,
    min_similarity: float = 0.5,
) -> tuple[AlignmentResult, int]:
    """Align tab chords with automatic transposition detection.

    Tries all 12 possible transpositions of the tab chords and returns
    the alignment with the best score. This handles cases where tabs
    are written in a different key than the recording.

    Parameters
    ----------
    tab_chords : list[TabChord]
        Chords extracted from a tab sheet.
    timed_chords : list[TimedChord]
        Timed chord annotations from audio analysis.
    lookahead : int
        Number of timed chords to search ahead (default 5).
    min_similarity : float
        Minimum similarity to accept a match (default 0.5).

    Returns
    -------
    tuple[AlignmentResult, int]
        Best alignment result and the transposition in semitones
        (0 = no transposition, positive = up, negative not used).

    Examples
    --------
    >>> # If tab is in G but audio is in A, transposition will be 2
    >>> result, transposition = align_with_transposition(tab_chords, timed_chords)
    >>> print(f"Best transposition: {transposition} semitones")
    """
    best_result: AlignmentResult | None = None
    best_transposition = 0
    best_score = -1.0

    for semitones in range(12):
        # Transpose tab chords
        if semitones == 0:
            transposed_tab = tab_chords
        else:
            transposed_tab = [
                TabChord(
                    chord=transpose_chord(tc.chord, semitones),
                    label=tc.label,
                    section=tc.section,
                    index=tc.index,
                )
                for tc in tab_chords
            ]

        # Run alignment
        result = align_chords(
            tab_chords=transposed_tab,
            timed_chords=timed_chords,
            lookahead=lookahead,
            min_similarity=min_similarity,
        )

        # Score = number of alignments / total tab chords (coverage)
        # with tie-breaker on normalized distance
        coverage = len(result.alignments) / len(tab_chords) if tab_chords else 0
        score = coverage - result.normalized_distance * 0.01  # Small penalty for distance

        if score > best_score:
            best_score = score
            best_result = result
            best_transposition = semitones

    if best_result is None:
        # Fallback to no transposition
        best_result = align_chords(tab_chords, timed_chords, lookahead, min_similarity)

    return best_result, best_transposition


def compute_alignment_metrics(
    result: AlignmentResult,
    similarity_fn: Callable[[Chord | None, Chord | None], float] | None = None,
) -> dict[str, float]:
    """Compute detailed alignment metrics.

    Parameters
    ----------
    result : AlignmentResult
        The alignment result to analyze.
    similarity_fn : Callable | None
        Optional similarity function. Defaults to chord_similarity.

    Returns
    -------
    dict[str, float]
        Dictionary with metrics:
        - coverage: fraction of tab chords aligned
        - avg_similarity: average similarity of aligned pairs
        - exact_matches: fraction of exact matches
        - root_matches: fraction of root-only matches
        - mismatches: fraction of mismatches (sim < 0.5)
    """
    if similarity_fn is None:
        similarity_fn = chord_similarity

    if not result.alignments:
        return {
            "coverage": 0.0,
            "avg_similarity": 0.0,
            "exact_matches": 0.0,
            "root_matches": 0.0,
            "mismatches": 1.0,
        }

    total = len(result.alignments)
    similarities = [
        similarity_fn(a.tab_chord.chord, a.timed_chord.chord)
        for a in result.alignments
    ]

    exact = sum(1 for s in similarities if s >= 1.0)
    root_only = sum(1 for s in similarities if 0.5 <= s < 1.0)
    mismatch = sum(1 for s in similarities if s < 0.5)

    return {
        "coverage": total / len(result.tab_chords) if result.tab_chords else 0.0,
        "avg_similarity": sum(similarities) / total,
        "exact_matches": exact / total,
        "root_matches": root_only / total,
        "mismatches": mismatch / total,
    }
