"""HMM training for chord alignment.

This module provides functions to train HMM parameters from labeled data
(audio files with ground truth chord annotations).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from decibel.audio_tab_aligner.hmm_parameters import HMMParameters
from decibel.music_objects.chord_alphabet import ChordAlphabet
from decibel.music_objects.chord_vocabulary import ChordVocabulary

from chord_parser.decibel.features import extract_audio_features

if TYPE_CHECKING:
    from chord_parser.alignment.models import TimedChord


def _align_chords_to_beats(
    beat_times: np.ndarray,
    gt_chords: list[TimedChord],
) -> list[str]:
    """Assign ground truth chords to beats by longest overlap.

    Parameters
    ----------
    beat_times : np.ndarray
        Array of beat times in seconds.
    gt_chords : list[TimedChord]
        Ground truth chord annotations with timing.

    Returns
    -------
    list[str]
        Chord label (Harte notation) for each beat.
    """
    nr_beats = len(beat_times) - 1  # Beats are intervals
    beat_chords: list[str] = []

    for i in range(nr_beats):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]

        # Find chord with longest overlap in this beat
        longest_duration = 0.0
        longest_chord = "N"

        for tc in gt_chords:
            chord_end = tc.end if tc.end is not None else beat_end + 1

            # Check overlap
            if tc.start < beat_end and chord_end > beat_start:
                overlap_start = max(tc.start, beat_start)
                overlap_end = min(chord_end, beat_end)
                duration = overlap_end - overlap_start

                if duration > longest_duration:
                    longest_duration = duration
                    # Convert to Harte string
                    if tc.chord is None:
                        longest_chord = "N"
                    else:
                        longest_chord = tc.chord.to_harte()

        beat_chords.append(longest_chord)

    return beat_chords


def _chord_to_alphabet_index(chord_str: str, alphabet: ChordAlphabet) -> int:
    """Convert a Harte chord string to alphabet index.

    Parameters
    ----------
    chord_str : str
        Chord in Harte notation (e.g., "G:min7", "C:maj").
    alphabet : ChordAlphabet
        The chord alphabet.

    Returns
    -------
    int
        Index in the alphabet, or 0 for unknown/no-chord.
    """
    if chord_str == "N":
        return 0

    # Try to match to MajMin vocabulary
    # Extract root and determine if major or minor
    from decibel.music_objects.chord import Chord as DECIBELChord

    try:
        chord = DECIBELChord.from_harte_chord_string(chord_str)
        if chord is None:
            return 0
        return int(alphabet.get_index_of_chord_in_alphabet(chord))
    except (KeyError, ValueError, AttributeError):
        # Chord not in vocabulary or failed to parse, return N
        return 0


def train_hmm(
    training_data: list[tuple[str | Path, list[TimedChord]]],
    vocabulary: ChordVocabulary | None = None,
    sampling_rate: int = 22050,
    hop_length: int = 256,
) -> HMMParameters:
    """Train HMM parameters from labeled audio data.

    Parameters
    ----------
    training_data : list[tuple[str | Path, list[TimedChord]]]
        List of (audio_path, ground_truth_chords) tuples.
    vocabulary : ChordVocabulary | None, optional
        Chord vocabulary to use. If None, uses MajMin.
    sampling_rate : int, optional
        Sampling rate for audio loading, by default 22050.
    hop_length : int, optional
        Hop length for feature extraction, by default 256.

    Returns
    -------
    HMMParameters
        Trained HMM parameters.
    """
    if vocabulary is None:
        vocabulary = ChordVocabulary.generate_chroma_major_minor()

    alphabet = ChordAlphabet(vocabulary)
    alphabet_size = len(alphabet.alphabet_list)

    # Collect chroma vectors per chord class
    chroma_per_chord: list[list[np.ndarray]] = [[] for _ in range(alphabet_size)]

    # Count transitions
    trans_counts = np.ones((alphabet_size, alphabet_size))  # Laplace smoothing
    init_counts = np.ones(alphabet_size)  # Laplace smoothing

    print(f"Training on {len(training_data)} songs...")

    for audio_path, gt_chords in training_data:
        print(f"  Processing {audio_path}...")

        # Extract audio features
        features = extract_audio_features(
            audio_path,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
        )

        # Align ground truth to beats
        beat_chords = _align_chords_to_beats(features.beat_times, gt_chords)

        # Convert to indices
        chord_indices = [
            _chord_to_alphabet_index(c, alphabet) for c in beat_chords
        ]

        # Collect chroma per chord
        for i, idx in enumerate(chord_indices):
            if i < len(features.beat_chroma):
                chroma_per_chord[idx].append(features.beat_chroma[i])

        # Count initial chord
        if chord_indices:
            init_counts[chord_indices[0]] += 1

        # Count transitions
        for i in range(len(chord_indices) - 1):
            trans_counts[chord_indices[i], chord_indices[i + 1]] += 1

    # Normalize transition matrix
    trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    # Normalize initial distribution
    init = init_counts / init_counts.sum()

    # Compute emission parameters (mean and covariance per chord)
    obs_mu = np.zeros((alphabet_size, 12))
    obs_sigma = np.zeros((alphabet_size, 12, 12))

    for i in range(alphabet_size):
        if len(chroma_per_chord[i]) > 1:
            chroma_matrix = np.array(chroma_per_chord[i])
            obs_mu[i] = np.mean(chroma_matrix, axis=0)
            obs_sigma[i] = np.cov(chroma_matrix.T, ddof=0)
        else:
            # Not enough data, use uniform/identity
            obs_mu[i] = np.ones(12) / 12
            obs_sigma[i] = np.eye(12) * 0.1

    # Precompute for emission probability calculation
    twelve_log_two_pi = 12 * np.log(2 * np.pi)
    log_det_sigma = np.zeros(alphabet_size)
    sigma_inverse = np.zeros((alphabet_size, 12, 12))

    for i in range(alphabet_size):
        det = np.linalg.det(obs_sigma[i])
        if det > 1e-10:
            log_det_sigma[i] = np.log(det)
            sigma_inverse[i] = np.linalg.pinv(obs_sigma[i])
        else:
            # Singular matrix, use regularized version
            obs_sigma[i] = obs_sigma[i] + np.eye(12) * 0.01
            log_det_sigma[i] = np.log(np.linalg.det(obs_sigma[i]))
            sigma_inverse[i] = np.linalg.pinv(obs_sigma[i])

    print(f"Training complete. Alphabet size: {alphabet_size}")

    # Report data per chord
    for i, name in enumerate(alphabet.alphabet_list):
        n = len(chroma_per_chord[i])
        if n > 0:
            print(f"  {name:6s}: {n:5d} samples")

    return HMMParameters(
        alphabet=alphabet,
        trans=trans,
        init=init,
        obs_mu=obs_mu,
        obs_sigma=obs_sigma,
        log_det_sigma=log_det_sigma,
        sigma_inverse=sigma_inverse,
        twelve_log_two_pi=twelve_log_two_pi,
        trained_on_keys=[],
    )


def save_hmm_parameters(hmm: HMMParameters, path: str | Path) -> None:
    """Save trained HMM parameters to a .npz file.

    Parameters
    ----------
    hmm : HMMParameters
        The trained HMM parameters.
    path : str | Path
        Path to save the parameters to.
    """
    path = Path(path)

    # Save vocabulary chord names for reconstruction
    chord_names = [str(c) for c in hmm.alphabet.alphabet_list]

    np.savez(
        path,
        trans=hmm.trans,
        init=hmm.init,
        obs_mu=hmm.obs_mu,
        obs_sigma=hmm.obs_sigma,
        log_det_sigma=hmm.log_det_sigma,
        sigma_inverse=hmm.sigma_inverse,
        twelve_log_two_pi=np.array([hmm.twelve_log_two_pi]),
        trained_on_keys=np.array(hmm.trained_on_keys),
        chord_names=np.array(chord_names),
    )
    print(f"Saved HMM parameters to {path}")


def load_hmm_parameters(path: str | Path) -> HMMParameters:
    """Load trained HMM parameters from a .npz file.

    Parameters
    ----------
    path : str | Path
        Path to the saved parameters.

    Returns
    -------
    HMMParameters
        The loaded HMM parameters.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    # Reconstruct vocabulary from saved chord names
    vocabulary = ChordVocabulary.generate_chroma_major_minor()
    alphabet = ChordAlphabet(vocabulary)

    return HMMParameters(
        alphabet=alphabet,
        trans=data["trans"],
        init=data["init"],
        obs_mu=data["obs_mu"],
        obs_sigma=data["obs_sigma"],
        log_det_sigma=data["log_det_sigma"],
        sigma_inverse=data["sigma_inverse"],
        twelve_log_two_pi=float(data["twelve_log_two_pi"][0]),
        trained_on_keys=list(data["trained_on_keys"]),
    )


def load_gt_chords_from_json(json_path: str | Path) -> list[TimedChord]:
    """Load ground truth chords from JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to JSON file with chord annotations.

    Returns
    -------
    list[TimedChord]
        List of timed chord annotations.
    """
    import json

    from chord_parser.alignment.models import TimedChord
    from chord_parser.converter import from_harte

    with open(json_path) as f:
        data = json.load(f)

    chords: list[TimedChord] = []
    for item in data:
        chord_str = item["chord"]
        if chord_str == "N":
            chord = None
        else:
            chord = from_harte(chord_str)

        chords.append(TimedChord(
            chord=chord,
            label=chord_str,
            start=item["start"],
            end=item["end"],
        ))

    return chords


def load_billboard_chords(json_path: str | Path) -> list[TimedChord]:
    """Load chord annotations from Billboard dataset format.

    Parameters
    ----------
    json_path : str | Path
        Path to Billboard JSON chord file.

    Returns
    -------
    list[TimedChord]
        List of timed chord annotations.
    """
    import json

    from chord_parser.alignment.models import TimedChord
    from chord_parser.converter import from_harte

    with open(json_path) as f:
        data = json.load(f)

    chords: list[TimedChord] = []
    for item in data["chords"]:
        chord_str = item["chord"]
        if chord_str == "N":
            chord = None
        else:
            try:
                chord = from_harte(chord_str)
            except (ValueError, KeyError):
                # Some complex chord symbols may not parse
                chord = None

        chords.append(TimedChord(
            chord=chord,
            label=chord_str,
            start=item["onset"],
            end=item["offset"],
        ))

    return chords


def load_billboard_dataset(
    dataset_dir: str | Path,
) -> list[tuple[Path, list[TimedChord]]]:
    """Load the Billboard dataset from a directory.

    Parameters
    ----------
    dataset_dir : str | Path
        Path to the Billboard dataset directory containing manifest.json.

    Returns
    -------
    list[tuple[Path, list[TimedChord]]]
        List of (audio_path, chord_annotations) tuples.
    """
    import json

    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    training_data: list[tuple[Path, list[TimedChord]]] = []

    for entry in manifest:
        audio_path = dataset_dir / entry["audio"]
        chord_path = dataset_dir / entry["chords"]

        if not audio_path.exists():
            print(f"  Skipping {entry['id']}: audio file not found")
            continue

        if not chord_path.exists():
            print(f"  Skipping {entry['id']}: chord file not found")
            continue

        gt_chords = load_billboard_chords(chord_path)
        training_data.append((audio_path, gt_chords))

    return training_data
