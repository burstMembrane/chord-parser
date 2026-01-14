"""Madmom chord recognition extractor.

This module provides chord detection using madmom's DeepChroma-based
chord recognition, which uses deep learning for chroma extraction
and a CRF for chord sequence decoding.

References
----------
Filip Korzeniowski and Gerhard Widmer,
"Feature Learning for Chord Recognition: The Deep Chroma Extractor",
Proceedings of the 17th International Society for Music Information
Retrieval Conference (ISMIR), 2016.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chord_parser.alignment.models import TimedChord
from chord_parser.converter import from_pychord

if TYPE_CHECKING:
    from numpy.typing import NDArray


def parse_madmom_chord(label: str) -> tuple[str, "from chord_parser.models import Chord | None"]:
    """Parse a madmom chord label to a Chord object.

    Madmom outputs labels in the format "Root:quality" (e.g., "G:min", "C:maj").

    Parameters
    ----------
    label : str
        Chord label from madmom (e.g., "G:min", "C:maj", "N").

    Returns
    -------
    tuple[str, Chord | None]
        Tuple of (normalized_label, parsed_chord).
        Returns (label, None) if label is "N" (no chord).

    Raises
    ------
    ValueError
        If the chord cannot be parsed.
    """
    from chord_parser.models import Chord

    if label == "N":
        return label, None

    # Madmom uses "Root:quality" format
    # Convert to pychord-compatible format
    if ":" in label:
        root, quality = label.split(":", 1)
        # Map madmom qualities to pychord qualities
        quality_map = {
            "maj": "",
            "min": "m",
            "dim": "dim",
            "aug": "aug",
            "maj7": "M7",
            "min7": "m7",
            "7": "7",
            "dim7": "dim7",
            "hdim7": "m7-5",  # half-diminished
            "sus4": "sus4",
            "sus2": "sus2",
        }
        pychord_quality = quality_map.get(quality, quality)
        pychord_label = f"{root}{pychord_quality}"
    else:
        pychord_label = label

    chord = from_pychord(pychord_label)
    return label, chord


def extract_chords_madmom(
    audio_path: str | Path,
    fps: int = 10,
) -> list[TimedChord]:
    """Extract timed chords from an audio file using madmom.

    Uses DeepChromaProcessor for feature extraction and
    DeepChromaChordRecognitionProcessor for chord decoding with CRF.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file (WAV, MP3, etc.).
    fps : int
        Frames per second for processing. Default is 10.

    Returns
    -------
    list[TimedChord]
        List of timed chord annotations with start/end times.

    Examples
    --------
    >>> chords = extract_chords_madmom("audio.wav")
    >>> for tc in chords[:3]:
    ...     print(f"{tc.start:.2f}-{tc.end:.2f}: {tc.label}")
    0.00-1.60: G:min
    1.60-3.20: C:maj
    3.20-4.80: F:maj
    """
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    from madmom.processors import SequentialProcessor

    audio_path = Path(audio_path)

    # Create processing pipeline
    chroma_processor = DeepChromaProcessor()
    chord_processor = DeepChromaChordRecognitionProcessor()
    pipeline = SequentialProcessor([chroma_processor, chord_processor])

    # Process audio file
    # Returns structured array with (start, end, label) tuples
    results: NDArray = pipeline(str(audio_path))

    # Convert to TimedChord objects
    timed_chords: list[TimedChord] = []
    for i, (start, end, label) in enumerate(results):
        try:
            norm_label, chord = parse_madmom_chord(label)
        except ValueError:
            # Skip unparseable chords
            norm_label = label
            chord = None

        timed_chords.append(
            TimedChord(
                chord=chord,
                label=norm_label,
                start=float(start),
                end=float(end),
            )
        )

    return timed_chords
