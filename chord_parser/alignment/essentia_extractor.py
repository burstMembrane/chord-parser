"""Essentia chord detection extractor.

This module provides chord detection using Essentia's ChordsDetection algorithm,
which uses template matching on HPCP (Harmonic Pitch Class Profile) chroma features.

Essentia's approach is a fast, template-based method that detects major and minor
chords by comparing HPCP vectors against ideal chord templates.
"""

from __future__ import annotations

from pathlib import Path

from chord_parser.alignment.models import TimedChord
from chord_parser.converter import from_pychord


def parse_essentia_chord(label: str) -> tuple[str, "from chord_parser.models import Chord | None"]:
    """Parse an Essentia chord label to a Chord object.

    Essentia outputs labels like "A", "Am", "A#", "Bbm", etc.

    Parameters
    ----------
    label : str
        Chord label from Essentia (e.g., "G", "Gm", "C#", "Dbm").

    Returns
    -------
    tuple[str, Chord | None]
        Tuple of (normalized_label, parsed_chord).
        Returns (label, None) if label is empty or unparseable.
    """
    if not label or label == "":
        return "N", None

    try:
        chord = from_pychord(label)
        return label, chord
    except ValueError:
        return label, None


def extract_chords_essentia(
    audio_path: str | Path,
    frame_size: int = 4096,
    hop_size: int = 2048,
    sample_rate: float = 44100.0,
    window_size: float = 2.0,
) -> list[TimedChord]:
    """Extract timed chords from an audio file using Essentia.

    Uses HPCP (Harmonic Pitch Class Profile) features with template-based
    chord detection for major/minor chord recognition.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file (WAV, MP3, etc.).
    frame_size : int
        Size of each analysis frame in samples. Default is 4096.
    hop_size : int
        Hop size between frames in samples. Default is 2048.
    sample_rate : float
        Expected sample rate. Audio will be resampled if different.
        Default is 44100.0.
    window_size : float
        Window size in seconds for chord detection smoothing.
        Default is 2.0 seconds.

    Returns
    -------
    list[TimedChord]
        List of timed chord annotations with start/end times.

    Examples
    --------
    >>> chords = extract_chords_essentia("audio.wav")
    >>> for tc in chords[:3]:
    ...     print(f"{tc.start:.2f}-{tc.end:.2f}: {tc.label}")
    0.00-1.50: Gm
    1.50-3.00: C
    3.00-4.50: F
    """
    import essentia
    import essentia.standard as es
    import numpy as np

    audio_path = Path(audio_path)

    # Load audio as mono
    loader = es.MonoLoader(filename=str(audio_path), sampleRate=sample_rate)
    audio = loader()

    # Initialize frame-level processors
    frame_cutter = es.FrameCutter(frameSize=frame_size, hopSize=hop_size, silentFrames="noise")
    windowing = es.Windowing(type="blackmanharris62")
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(
        orderBy="magnitude",
        magnitudeThreshold=0.00001,
        minFrequency=20,
        maxFrequency=3500,
        maxPeaks=60,
    )
    hpcp = es.HPCP()

    # Compute HPCP for all frames
    hpcp_frames = []
    frame_cutter.configure(frameSize=frame_size, hopSize=hop_size)

    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        windowed = windowing(frame)
        spec = spectrum(windowed)
        freqs, mags = spectral_peaks(spec)
        hpcp_vector = hpcp(freqs, mags)
        hpcp_frames.append(hpcp_vector)

    if not hpcp_frames:
        return []

    hpcp_matrix = np.array(hpcp_frames)

    # Detect chords from HPCP matrix
    # windowSize is in HPCP frames, so convert from seconds
    frames_per_second = sample_rate / hop_size
    window_frames = int(window_size * frames_per_second)

    chord_detection = es.ChordsDetection(hopSize=hop_size, windowSize=window_frames)
    chord_labels, chord_strengths = chord_detection(hpcp_matrix)

    # Convert frame indices to timestamps and merge consecutive identical chords
    time_per_frame = hop_size / sample_rate
    timed_chords: list[TimedChord] = []

    if len(chord_labels) == 0:
        return []

    current_label = chord_labels[0]
    current_start = 0.0

    for i in range(1, len(chord_labels)):
        if chord_labels[i] != current_label:
            # End current chord segment
            end_time = i * time_per_frame
            norm_label, chord = parse_essentia_chord(current_label)

            timed_chords.append(
                TimedChord(
                    chord=chord,
                    label=norm_label,
                    start=current_start,
                    end=end_time,
                )
            )

            # Start new segment
            current_label = chord_labels[i]
            current_start = end_time

    # Add final segment
    end_time = len(chord_labels) * time_per_frame
    norm_label, chord = parse_essentia_chord(current_label)
    timed_chords.append(
        TimedChord(
            chord=chord,
            label=norm_label,
            start=current_start,
            end=end_time,
        )
    )

    return timed_chords
