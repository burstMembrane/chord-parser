"""Audio feature extraction for HMM-based chord alignment.

This module provides beat-synchronous chroma feature extraction,
wrapping DECIBEL's feature extraction with a clean interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from decibel.audio_tab_aligner.feature_extractor import get_audio_features


@dataclass(frozen=True)
class AudioFeatures:
    """Beat-synchronous audio features for chord alignment.

    Parameters
    ----------
    beat_times : np.ndarray
        Array of beat times in seconds, shape (nr_beats,).
    beat_chroma : np.ndarray
        Chroma features per beat, shape (nr_beats, 12).
    """

    beat_times: np.ndarray
    beat_chroma: np.ndarray

    @property
    def nr_beats(self) -> int:
        """Return the number of beats."""
        return len(self.beat_times)


def extract_audio_features(
    audio_path: str | Path,
    *,
    sampling_rate: int = 22050,
    hop_length: int = 256,
) -> AudioFeatures:
    """Extract beat-synchronous chroma features from an audio file.

    Uses librosa for audio loading, HPSS separation, beat tracking,
    and chroma-CQT extraction. Features are beat-synchronized by
    averaging chroma between consecutive beat positions.

    Parameters
    ----------
    audio_path : str | Path
        Path to the audio file.
    sampling_rate : int, optional
        Sampling rate for audio loading, by default 22050.
    hop_length : int, optional
        Hop length for feature extraction, by default 256.

    Returns
    -------
    AudioFeatures
        Beat-synchronous chroma features.

    Notes
    -----
    Requires librosa to be installed (available via the 'audio' extra).
    """
    audio_path_str = str(audio_path)
    beat_times, beat_chroma = get_audio_features(
        audio_path_str,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
    )
    return AudioFeatures(beat_times=beat_times, beat_chroma=beat_chroma)
