"""Pitch class operations for chord comparison.

This module provides pitch class (0-11) representations of chords
for similarity computation based on actual note content rather than
just chord name matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chord_parser.models import Chord

# Note name to pitch class (0-11, where C=0)
NOTE_TO_PC: dict[str, int] = {
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

# Interval name to semitones from root
INTERVAL_TO_SEMITONES: dict[str, int] = {
    "1": 0,
    "b2": 1,
    "2": 2,
    "#2": 3,
    "b3": 3,
    "3": 4,
    "#3": 5,
    "4": 5,
    "#4": 6,
    "b5": 6,
    "5": 7,
    "#5": 8,
    "b6": 8,
    "6": 9,
    "#6": 10,
    "bb7": 9,
    "b7": 10,
    "7": 11,
    "#7": 0,
    "9": 2,  # 9th = 2nd + octave
    "b9": 1,
    "#9": 3,
    "11": 5,  # 11th = 4th + octave
    "#11": 6,
    "13": 9,  # 13th = 6th + octave
    "b13": 8,
}

# Harte quality to list of intervals (following DECIBEL's approach)
QUALITY_TO_INTERVALS: dict[str, list[str]] = {
    # Triads
    "maj": ["1", "3", "5"],
    "min": ["1", "b3", "5"],
    "dim": ["1", "b3", "b5"],
    "aug": ["1", "3", "#5"],
    # Suspended
    "sus2": ["1", "2", "5"],
    "sus4": ["1", "4", "5"],
    # Sixth chords
    "maj6": ["1", "3", "5", "6"],
    "min6": ["1", "b3", "5", "6"],
    # Seventh chords
    "7": ["1", "3", "5", "b7"],
    "maj7": ["1", "3", "5", "7"],
    "min7": ["1", "b3", "5", "b7"],
    "dim7": ["1", "b3", "b5", "bb7"],
    "hdim7": ["1", "b3", "b5", "b7"],
    "minmaj7": ["1", "b3", "5", "7"],
    "aug7": ["1", "3", "#5", "b7"],
    # Suspended seventh
    "7sus4": ["1", "4", "5", "b7"],
    "7sus2": ["1", "2", "5", "b7"],
    "sus4(b7)": ["1", "4", "5", "b7"],
    "sus2(b7)": ["1", "2", "5", "b7"],
    # Ninth chords
    "9": ["1", "3", "5", "b7", "9"],
    "maj9": ["1", "3", "5", "7", "9"],
    "min9": ["1", "b3", "5", "b7", "9"],
    "maj(9)": ["1", "3", "5", "9"],
    "min(9)": ["1", "b3", "5", "9"],
    # Eleventh chords
    "11": ["1", "3", "5", "b7", "9", "11"],
    "maj11": ["1", "3", "5", "7", "9", "11"],
    "min11": ["1", "b3", "5", "b7", "9", "11"],
    # Thirteenth chords
    "13": ["1", "3", "5", "b7", "9", "13"],
    "maj13": ["1", "3", "5", "7", "9", "13"],
    "min13": ["1", "b3", "5", "b7", "9", "13"],
    # Power chord
    "5": ["1", "5"],
    # Dim6 (enharmonic with dim7 in some contexts)
    "dim6": ["1", "b3", "b5", "6"],
}


def note_to_pc(note: str) -> int:
    """Convert a note name to pitch class (0-11).

    Parameters
    ----------
    note : str
        Note name (e.g., "C", "F#", "Bb").

    Returns
    -------
    int
        Pitch class (0-11, where C=0).

    Raises
    ------
    ValueError
        If the note name is not recognized.

    Examples
    --------
    >>> note_to_pc("C")
    0
    >>> note_to_pc("F#")
    6
    >>> note_to_pc("Bb")
    10
    """
    if note in NOTE_TO_PC:
        return NOTE_TO_PC[note]
    msg = f"Unknown note: {note}"
    raise ValueError(msg)


def chord_to_pitch_classes(chord: Chord) -> frozenset[int]:
    """Convert a Chord to a set of pitch classes.

    Parameters
    ----------
    chord : Chord
        The chord to convert.

    Returns
    -------
    frozenset[int]
        Set of pitch classes (0-11) in the chord.

    Examples
    --------
    >>> from chord_parser.models import Chord
    >>> chord = Chord(root="C", quality="maj")
    >>> sorted(chord_to_pitch_classes(chord))
    [0, 4, 7]
    >>> chord = Chord(root="G", quality="min")
    >>> sorted(chord_to_pitch_classes(chord))
    [2, 7, 10]
    """
    root_pc = note_to_pc(chord.root)

    # Get intervals for this quality
    intervals = QUALITY_TO_INTERVALS.get(chord.quality, ["1", "3", "5"])

    # Convert intervals to pitch classes
    pitch_classes: set[int] = set()
    for interval in intervals:
        if interval in INTERVAL_TO_SEMITONES:
            pc = (root_pc + INTERVAL_TO_SEMITONES[interval]) % 12
            pitch_classes.add(pc)

    # Add bass note if present
    if chord.bass:
        try:
            bass_pc = note_to_pc(chord.bass)
            pitch_classes.add(bass_pc)
        except ValueError:
            pass

    return frozenset(pitch_classes)


def pitch_class_jaccard(pc1: frozenset[int], pc2: frozenset[int]) -> float:
    """Compute Jaccard similarity between two pitch class sets.

    Parameters
    ----------
    pc1 : frozenset[int]
        First set of pitch classes.
    pc2 : frozenset[int]
        Second set of pitch classes.

    Returns
    -------
    float
        Jaccard similarity (0.0 to 1.0).

    Examples
    --------
    >>> pitch_class_jaccard(frozenset({0, 4, 7}), frozenset({0, 4, 7}))
    1.0
    >>> pitch_class_jaccard(frozenset({0, 4, 7}), frozenset({0, 3, 7}))
    0.5
    """
    if not pc1 or not pc2:
        return 0.0
    intersection = len(pc1 & pc2)
    union = len(pc1 | pc2)
    return intersection / union if union > 0 else 0.0


def chord_pitch_similarity(chord1: Chord | None, chord2: Chord | None) -> float:
    """Compute pitch class Jaccard similarity between two chords.

    Parameters
    ----------
    chord1 : Chord | None
        First chord.
    chord2 : Chord | None
        Second chord.

    Returns
    -------
    float
        Jaccard similarity of pitch class sets (0.0 to 1.0).

    Examples
    --------
    >>> from chord_parser.models import Chord
    >>> c1 = Chord(root="C", quality="maj")
    >>> c2 = Chord(root="C", quality="maj")
    >>> chord_pitch_similarity(c1, c2)
    1.0
    >>> c3 = Chord(root="C", quality="min")
    >>> chord_pitch_similarity(c1, c3)  # C vs Cm share C and G
    0.5
    """
    if chord1 is None or chord2 is None:
        return 0.0

    pc1 = chord_to_pitch_classes(chord1)
    pc2 = chord_to_pitch_classes(chord2)
    return pitch_class_jaccard(pc1, pc2)


def transpose_chord(chord: Chord, semitones: int) -> Chord:
    """Transpose a chord by a number of semitones.

    Parameters
    ----------
    chord : Chord
        The chord to transpose.
    semitones : int
        Number of semitones to transpose (positive = up).

    Returns
    -------
    Chord
        Transposed chord.

    Examples
    --------
    >>> from chord_parser.models import Chord
    >>> c = Chord(root="C", quality="maj")
    >>> transpose_chord(c, 2).root
    'D'
    >>> transpose_chord(c, -1).root
    'B'
    """
    from chord_parser.models import Chord as ChordModel

    # Pitch class to sharp note name (prefer sharps for consistency)
    PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    root_pc = note_to_pc(chord.root)
    new_root_pc = (root_pc + semitones) % 12
    new_root = PC_TO_NOTE[new_root_pc]

    new_bass = None
    if chord.bass:
        try:
            bass_pc = note_to_pc(chord.bass)
            new_bass_pc = (bass_pc + semitones) % 12
            new_bass = PC_TO_NOTE[new_bass_pc]
        except ValueError:
            new_bass = chord.bass

    return ChordModel(root=new_root, quality=chord.quality, bass=new_bass)


def roots_match(chord1: Chord | None, chord2: Chord | None) -> bool:
    """Check if two chords have the same root (enharmonic equivalence).

    Parameters
    ----------
    chord1 : Chord | None
        First chord.
    chord2 : Chord | None
        Second chord.

    Returns
    -------
    bool
        True if roots are enharmonically equivalent.

    Examples
    --------
    >>> from chord_parser.models import Chord
    >>> c1 = Chord(root="C#", quality="maj")
    >>> c2 = Chord(root="Db", quality="min")
    >>> roots_match(c1, c2)
    True
    """
    if chord1 is None or chord2 is None:
        return False
    try:
        return note_to_pc(chord1.root) == note_to_pc(chord2.root)
    except ValueError:
        return False


def quality_category(quality: str) -> str:
    """Get the category of a chord quality.

    Categories: "major", "minor", "diminished", "augmented", "suspended", "other"

    Parameters
    ----------
    quality : str
        Harte quality string.

    Returns
    -------
    str
        Quality category.

    Examples
    --------
    >>> quality_category("maj")
    'major'
    >>> quality_category("min7")
    'minor'
    >>> quality_category("dim")
    'diminished'
    """
    if quality in ("maj", "maj7", "maj6", "maj9", "maj11", "maj13", "maj(9)", "7", "9", "11", "13"):
        return "major"
    if quality in ("min", "min7", "min6", "min9", "min11", "min13", "min(9)", "minmaj7"):
        return "minor"
    if quality in ("dim", "dim7", "dim6", "hdim7"):
        return "diminished"
    if quality in ("aug", "aug7"):
        return "augmented"
    if quality in ("sus2", "sus4", "7sus2", "7sus4", "sus2(b7)", "sus4(b7)"):
        return "suspended"
    return "other"


def weighted_chord_similarity(
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
    If root_gate=True (default), mismatched roots return 0.0 regardless
    of other weights.

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

    Examples
    --------
    >>> from chord_parser.models import Chord
    >>> c1 = Chord(root="G", quality="min7")
    >>> c2 = Chord(root="G", quality="min7")
    >>> weighted_chord_similarity(c1, c2)
    1.0
    >>> c3 = Chord(root="G", quality="min")
    >>> weighted_chord_similarity(c1, c3)  # Same root, same category, different PCs
    0.9...
    """
    if chord1 is None or chord2 is None:
        return 0.0

    # Root match
    root_match = 1.0 if roots_match(chord1, chord2) else 0.0

    # If root gate is enabled and roots don't match, return 0
    if root_gate and root_match == 0.0:
        return 0.0

    # Pitch class Jaccard
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
