"""DTW-based chord alignment.

This module implements Dynamic Time Warping alignment between
tab chords and timed Chordino annotations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chord_parser.models import Chord

from chord_parser.alignment.models import (
    AlignedChord,
    AlignmentResult,
    TabChord,
    TimedChord,
)

# Type alias for distance functions
DistanceFn = Callable[["Chord | None", "Chord | None"], float]


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

    # Exact match on root, quality, and bass
    if chord1.root == chord2.root and chord1.quality == chord2.quality and chord1.bass == chord2.bass:
        return 0.0

    return 1.0


# Pitch class mapping (C=0, C#/Db=1, ..., B=11)
NOTE_TO_PITCH_CLASS: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0,
}


def note_to_pitch_class(note: str) -> int:
    """Convert a note name to pitch class (0-11).

    Parameters
    ----------
    note : str
        Note name (e.g., "C", "F#", "Bb").

    Returns
    -------
    int
        Pitch class (0-11).

    Raises
    ------
    ValueError
        If note is not recognized.
    """
    if note in NOTE_TO_PITCH_CLASS:
        return NOTE_TO_PITCH_CLASS[note]
    msg = f"Unknown note: {note}"
    raise ValueError(msg)


def chord_to_pitch_class_set(chord: Chord, include_bass: bool = False) -> frozenset[int]:
    """Convert a Chord to a set of pitch classes.

    Uses pychord to decompose the chord into component notes,
    then converts each note to its pitch class (0-11).

    Parameters
    ----------
    chord : Chord
        The chord to convert.
    include_bass : bool
        Whether to include the bass note for slash chords.
        Default is False (drop bass to avoid inflating overlap).

    Returns
    -------
    frozenset[int]
        Set of pitch classes (0-11).
    """
    from pychord import Chord as PyChord

    # Convert our Chord to pychord notation and get components
    pychord_str = chord.to_pychord()
    pc = PyChord(pychord_str)
    components = pc.components()

    # If not including bass and this is a slash chord, remove bass note
    if not include_bass and chord.bass:
        # pychord includes bass in components for slash chords
        # Remove it (it's usually the first element)
        bass_pc = note_to_pitch_class(chord.bass)
        pitch_classes = {note_to_pitch_class(n) for n in components}
        pitch_classes.discard(bass_pc)
        return frozenset(pitch_classes)

    return frozenset(note_to_pitch_class(n) for n in components)


def jaccard_similarity(set1: frozenset[int], set2: frozenset[int]) -> float:
    """Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set1 : frozenset[int]
        First set.
    set2 : frozenset[int]
        Second set.

    Returns
    -------
    float
        Jaccard similarity (0.0 to 1.0).
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def chord_distance_pitchclass(chord1: Chord | None, chord2: Chord | None) -> float:
    """Compute distance using pitch-class set Jaccard similarity.

    Uses weighted Jaccard on pitch-class sets with root penalty:
    - If roots match: dist = 1 - jaccard_similarity
    - If roots differ: dist = 0.6 + 0.4 * (1 - jaccard_similarity)

    This prevents wrong-root chords from accidentally looking good
    due to note overlap.

    Parameters
    ----------
    chord1 : Chord | None
        First chord (None represents "no chord").
    chord2 : Chord | None
        Second chord (None represents "no chord").

    Returns
    -------
    float
        Distance between 0.0 (identical) and 1.0 (no match).

    Examples
    --------
    >>> from chord_parser.models import Chord
    >>> chord_distance_pitchclass(Chord("A", "min"), Chord("A", "min7"))
    0.25  # Am={A,C,E}, Am7={A,C,E,G}, jaccard=3/4=0.75, dist=0.25
    >>> chord_distance_pitchclass(Chord("G", "min"), Chord("G", "min7"))
    0.25  # Same pattern
    >>> chord_distance_pitchclass(Chord("B", "dim"), Chord("B", "maj"))
    0.8   # Bdim={B,D,F}, B={B,D#,F#}, only B overlaps, jaccard=1/5=0.2
    """
    if chord1 is None and chord2 is None:
        return 0.0
    if chord1 is None or chord2 is None:
        return 1.0

    # Get pitch class sets (excluding bass notes)
    pc_set1 = chord_to_pitch_class_set(chord1, include_bass=False)
    pc_set2 = chord_to_pitch_class_set(chord2, include_bass=False)

    # Compute Jaccard similarity
    sim = jaccard_similarity(pc_set1, pc_set2)

    # Apply root weighting
    if chord1.root == chord2.root:
        # Same root: distance is just 1 - similarity
        return 1.0 - sim
    # Different roots: add penalty to prevent false matches
    # Even with perfect note overlap, different roots get 0.6 minimum distance
    return 0.6 + 0.4 * (1.0 - sim)


def chord_distance_flexible(chord1: Chord | None, chord2: Chord | None) -> float:  # noqa: PLR0911
    """Compute flexible distance between two chords.

    Uses a tiered matching approach:
    - 0.0: Exact match (root, quality, bass all identical)
    - 0.2: Same root, different quality (e.g., Gm vs Gm7, Bdim vs B)
    - 0.4: Bass note of one matches root of other (e.g., Adim7/G vs Gm)
    - 1.0: No match

    Parameters
    ----------
    chord1 : Chord | None
        First chord (None represents "no chord").
    chord2 : Chord | None
        Second chord (None represents "no chord").

    Returns
    -------
    float
        Distance between 0.0 (identical) and 1.0 (no match).
    """
    if chord1 is None and chord2 is None:
        return 0.0
    if chord1 is None or chord2 is None:
        return 1.0

    # Exact match
    if chord1.root == chord2.root and chord1.quality == chord2.quality and chord1.bass == chord2.bass:
        return 0.0

    # Same root, different quality (e.g., Gm vs Gm7, Bdim vs B)
    if chord1.root == chord2.root:
        return 0.2

    # Bass note of slash chord matches root of other chord
    # e.g., Adim7/G matches Gm because G is the bass
    if (chord1.bass is not None and chord1.bass == chord2.root) or (
        chord2.bass is not None and chord2.bass == chord1.root
    ):
        return 0.4

    return 1.0


def build_distance_matrix(
    tab_chords: list[TabChord],
    timed_chords: list[TimedChord],
    distance_fn: DistanceFn = chord_distance_flexible,
) -> NDArray[np.float64]:
    """Build a distance matrix between tab and timed chords.

    Parameters
    ----------
    tab_chords : list[TabChord]
        Chords from the tab.
    timed_chords : list[TimedChord]
        Timed chords from Chordino.
    distance_fn : DistanceFn
        Function to compute distance between two chords.
        Default is chord_distance_flexible.

    Returns
    -------
    NDArray[np.float64]
        Distance matrix of shape (len(tab_chords), len(timed_chords)).
    """
    n = len(tab_chords)
    m = len(timed_chords)
    matrix = np.zeros((n, m), dtype=np.float64)

    for i, tab_chord in enumerate(tab_chords):
        for j, timed_chord in enumerate(timed_chords):
            matrix[i, j] = distance_fn(tab_chord.chord, timed_chord.chord)

    return matrix


def align_chords(
    tab_chords: list[TabChord],
    timed_chords: list[TimedChord],
    step_pattern: str = "symmetric2",
    distance_fn: DistanceFn = chord_distance_flexible,
) -> AlignmentResult:
    """Align tab chords to timed chords using DTW.

    Parameters
    ----------
    tab_chords : list[TabChord]
        Chords extracted from a tab sheet.
    timed_chords : list[TimedChord]
        Timed chord annotations from Chordino.
    step_pattern : str
        DTW step pattern name. Default is "symmetric2".
    distance_fn : DistanceFn
        Function to compute distance between two chords.
        Default is chord_distance_flexible. Use chord_distance_exact
        for strict matching.

    Returns
    -------
    AlignmentResult
        The alignment result with matched chord pairs.

    Notes
    -----
    Uses the dtw-python library for DTW computation.
    The query sequence is the tab chords, and the reference
    sequence is the timed chords.
    """
    from dtw import dtw

    # Filter out "N" (no chord) from timed chords for alignment
    # We want to align actual chords, not silence markers
    filtered_timed = [tc for tc in timed_chords if tc.chord is not None]

    if not tab_chords or not filtered_timed:
        return AlignmentResult(
            alignments=(),
            tab_chords=tuple(tab_chords),
            timed_chords=tuple(timed_chords),
            total_distance=float("inf"),
            normalized_distance=float("inf"),
        )

    # Build distance matrix
    dist_matrix = build_distance_matrix(tab_chords, filtered_timed, distance_fn)

    # Run DTW
    alignment = dtw(
        dist_matrix,
        step_pattern=step_pattern,
        keep_internals=True,
    )

    # Extract alignment path
    # alignment.index1 contains query (tab) indices
    # alignment.index2 contains reference (timed) indices
    path_tab = alignment.index1
    path_timed = alignment.index2

    # Build aligned chord pairs
    # For each unique tab chord index, find its matched timed chord(s)
    aligned: list[AlignedChord] = []
    seen_tab_indices: set[int] = set()

    for tab_idx, timed_idx in zip(path_tab, path_timed, strict=True):
        if tab_idx not in seen_tab_indices:
            seen_tab_indices.add(tab_idx)
            tab_chord = tab_chords[tab_idx]
            timed_chord = filtered_timed[timed_idx]
            distance = dist_matrix[tab_idx, timed_idx]

            aligned.append(
                AlignedChord(
                    tab_chord=tab_chord,
                    timed_chord=timed_chord,
                    distance=distance,
                )
            )

    return AlignmentResult(
        alignments=tuple(aligned),
        tab_chords=tuple(tab_chords),
        timed_chords=tuple(timed_chords),
        total_distance=float(alignment.distance),
        normalized_distance=float(alignment.normalizedDistance),
    )
