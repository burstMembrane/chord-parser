"""HMM-based chord alignment using DECIBEL's jump alignment algorithm.

This module provides functions to align tab chords with audio using
a Hidden Markov Model with Viterbi decoding. It supports automatic
transposition detection by trying all 12 semitone shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from decibel.audio_tab_aligner.hmm_parameters import HMMParameters
from decibel.audio_tab_aligner.jump_alignment import (
    _calculate_altered_transition_matrix,
    _transpose_chord_label,
)
from decibel.music_objects.chord_alphabet import ChordAlphabet
from decibel.music_objects.chord_vocabulary import ChordVocabulary

from chord_parser.alignment.models import AlignedChord, AlignmentResult, TimedChord
from chord_parser.decibel.adapter import tabchords_to_alignment_arrays
from chord_parser.decibel.features import AudioFeatures, extract_audio_features

if TYPE_CHECKING:
    from pathlib import Path

    from chord_parser.alignment.models import TabChord

# Minimum number of chords required for alignment
MIN_CHORDS_FOR_ALIGNMENT = 2


@dataclass(frozen=True)
class HMMAlignmentResult:
    """Result of HMM-based alignment.

    Parameters
    ----------
    alignment_result : AlignmentResult
        Standard alignment result with matched chord pairs.
    log_likelihood : float
        Log-likelihood of the best alignment path.
    transposition : int
        Number of semitones the tab was transposed (0-11).
    """

    alignment_result: AlignmentResult
    log_likelihood: float
    transposition: int


def create_default_hmm_parameters(
    vocabulary: ChordVocabulary | None = None,
) -> HMMParameters:
    """Create HMM parameters using chord template-based emission probabilities.

    This creates a simplified HMM without training data, using the chord
    templates as emission probability means and uniform priors.

    Parameters
    ----------
    vocabulary : ChordVocabulary | None, optional
        Chord vocabulary to use. If None, uses MajMin (major/minor chords).

    Returns
    -------
    HMMParameters
        HMM parameters suitable for alignment.
    """
    if vocabulary is None:
        vocabulary = ChordVocabulary.generate_chroma_major_minor()

    alphabet = ChordAlphabet(vocabulary)
    alphabet_size = len(alphabet.alphabet_list)

    # Uniform transition probabilities with self-loop preference
    trans = np.ones((alphabet_size, alphabet_size))
    for i in range(alphabet_size):
        trans[i, i] = 5.0  # Prefer staying on same chord
    trans = trans / trans.sum(axis=1, keepdims=True)

    # Uniform initial distribution
    init = np.ones(alphabet_size) / alphabet_size

    # Use chord templates as emission means
    obs_mu = np.zeros((alphabet_size, 12))
    obs_mu[0] = np.ones(12) / 12  # "N" (no chord) has flat chroma

    for i, template in enumerate(vocabulary.chord_templates):
        obs_mu[i + 1] = np.array(template.chroma_list, dtype=float)
        # Normalize to sum to 1
        if obs_mu[i + 1].sum() > 0:
            obs_mu[i + 1] = obs_mu[i + 1] / obs_mu[i + 1].sum()

    # Use identity covariance (diagonal with small variance)
    obs_sigma = np.zeros((alphabet_size, 12, 12))
    for i in range(alphabet_size):
        obs_sigma[i] = np.eye(12) * 0.1

    # Precompute for emission probability calculation
    twelve_log_two_pi = 12 * np.log(2 * np.pi)
    log_det_sigma = np.zeros(alphabet_size)
    sigma_inverse = np.zeros((alphabet_size, 12, 12))

    for i in range(alphabet_size):
        log_det_sigma[i] = np.log(np.linalg.det(obs_sigma[i]))
        sigma_inverse[i] = np.linalg.pinv(obs_sigma[i])

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


def _compute_log_emission_matrix(
    features: np.ndarray,
    hmm_parameters: HMMParameters,
) -> np.ndarray:
    """Compute log emission probability matrix.

    Parameters
    ----------
    features : np.ndarray
        Chroma features, shape (nr_beats, 12).
    hmm_parameters : HMMParameters
        HMM parameters with emission distributions.

    Returns
    -------
    np.ndarray
        Log emission probabilities, shape (alphabet_size, nr_beats).
    """
    alphabet_size = len(hmm_parameters.alphabet.alphabet_list)
    nr_beats = features.shape[0]
    log_emission = np.zeros((alphabet_size, nr_beats))

    for i in range(alphabet_size):
        for b in range(nr_beats):
            diff = features[b] - hmm_parameters.obs_mu[i]
            diff_mat = np.asmatrix(diff)
            mahal = float(diff_mat @ hmm_parameters.sigma_inverse[i] @ diff_mat.T)
            log_emission[i, b] = -0.5 * (hmm_parameters.log_det_sigma[i] + mahal + hmm_parameters.twelve_log_two_pi)

    return log_emission


def _viterbi_with_transposition(
    nr_of_chords: int,
    chord_ids: np.ndarray,
    is_first_in_line: np.ndarray,
    is_last_in_line: np.ndarray,
    log_emission: np.ndarray,
    hmm_parameters: HMMParameters,
    p_f: float,
    p_b: float,
) -> tuple[np.ndarray, float, int]:
    """Run Viterbi algorithm with transposition search.

    Parameters
    ----------
    nr_of_chords : int
        Number of chords in the tab.
    chord_ids : np.ndarray
        Chord alphabet indices for each tab chord.
    is_first_in_line : np.ndarray
        Binary array indicating first chord in each section.
    is_last_in_line : np.ndarray
        Binary array indicating last chord in each section.
    log_emission : np.ndarray
        Log emission probabilities, shape (alphabet_size, nr_beats).
    hmm_parameters : HMMParameters
        HMM parameters.
    p_f : float
        Forward jump probability.
    p_b : float
        Backward jump probability.

    Returns
    -------
    tuple[np.ndarray, float, int]
        - viterbi_path: Best path through tab chords for each beat
        - best_likelihood: Log-likelihood of best path
        - best_transposition: Transposition in semitones (0-11)
    """
    nr_beats = log_emission.shape[1]
    best_transposition = 0
    best_likelihood = float("-inf")
    best_viterbi_path: np.ndarray | None = None

    for semitone in range(12):
        # Transpose chord IDs
        transposed_ids = np.array(
            [_transpose_chord_label(int(c), semitone, hmm_parameters.alphabet) for c in chord_ids]
        )

        # Calculate altered transition matrix
        altered_trans = _calculate_altered_transition_matrix(
            nr_of_chords,
            transposed_ids,
            is_first_in_line,
            is_last_in_line,
            hmm_parameters,
            p_f,
            p_b,
        )

        # Viterbi forward pass
        g = np.zeros((nr_beats, nr_of_chords))
        tr = np.zeros((nr_beats, nr_of_chords), dtype=int)

        # Initialize first beat
        for j in range(nr_of_chords):
            g[0, j] = log_emission[transposed_ids[j], 0] + np.log(hmm_parameters.init[transposed_ids[j]])

        # Forward pass
        for t in range(1, nr_beats):
            for j in range(nr_of_chords):
                best_prev = float("-inf")
                best_prev_chord = 0
                for c in range(nr_of_chords):
                    if altered_trans[c, j] > 0:
                        score = g[t - 1, c] + np.log(altered_trans[c, j])
                        if score > best_prev:
                            best_prev = score
                            best_prev_chord = c
                g[t, j] = log_emission[transposed_ids[j], t] + best_prev
                tr[t, j] = best_prev_chord

        # Find best final state
        log_likelihood = float("-inf")
        last_chord = 0
        for c in range(nr_of_chords):
            if g[-1, c] > log_likelihood:
                log_likelihood = g[-1, c]
                last_chord = c

        if log_likelihood > best_likelihood:
            best_likelihood = log_likelihood
            best_transposition = semitone

            # Backtrack to get Viterbi path
            path = [last_chord]
            for t in range(nr_beats - 1, 0, -1):
                path.append(tr[t, path[-1]])
            best_viterbi_path = np.array(list(reversed(path)))

    if best_viterbi_path is None:
        best_viterbi_path = np.zeros(nr_beats, dtype=int)

    return best_viterbi_path, best_likelihood, best_transposition


def _path_to_timed_chords(
    viterbi_path: np.ndarray,
    beat_times: np.ndarray,
    tab_chords: list[TabChord],
) -> list[TimedChord]:
    """Convert Viterbi path to timed chord sequence.

    Parameters
    ----------
    viterbi_path : np.ndarray
        Index into tab_chords for each beat.
    beat_times : np.ndarray
        Time in seconds for each beat.
    tab_chords : list[TabChord]
        Original tab chords.

    Returns
    -------
    list[TimedChord]
        Timed chord sequence with merged consecutive duplicates.
    """
    if len(viterbi_path) == 0:
        return []

    timed_chords: list[TimedChord] = []
    current_chord_idx = viterbi_path[0]
    start_time = 0.0

    for i in range(1, len(viterbi_path)):
        if viterbi_path[i] != current_chord_idx:
            # Chord changed, emit previous
            tc = tab_chords[current_chord_idx]
            timed_chords.append(
                TimedChord(
                    chord=tc.chord,
                    label=tc.label,
                    start=start_time,
                    end=float(beat_times[i - 1]) if i > 0 else float(beat_times[0]),
                    source_index=current_chord_idx,
                )
            )
            current_chord_idx = viterbi_path[i]
            start_time = float(beat_times[i - 1]) if i > 0 else 0.0

    # Emit final chord
    tc = tab_chords[current_chord_idx]
    end_time = float(beat_times[-1]) if len(beat_times) > 0 else None
    timed_chords.append(
        TimedChord(
            chord=tc.chord,
            label=tc.label,
            start=start_time,
            end=end_time,
            source_index=current_chord_idx,
        )
    )

    return timed_chords


def align_tab_with_chroma(
    tab_chords: list[TabChord],
    audio_features: AudioFeatures,
    *,
    hmm_parameters: HMMParameters | None = None,
    p_f: float = 0.05,
    p_b: float = 0.05,
) -> HMMAlignmentResult:
    """Align tab chords with pre-extracted audio features using HMM.

    Parameters
    ----------
    tab_chords : list[TabChord]
        Tab chords to align.
    audio_features : AudioFeatures
        Beat-synchronous chroma features from audio.
    hmm_parameters : HMMParameters | None, optional
        Pre-trained HMM parameters. If None, uses template-based defaults.
    p_f : float, optional
        Forward jump probability for line transitions, by default 0.05.
    p_b : float, optional
        Backward jump probability for line transitions, by default 0.05.

    Returns
    -------
    HMMAlignmentResult
        Alignment result with log-likelihood and transposition info.
    """
    if hmm_parameters is None:
        hmm_parameters = create_default_hmm_parameters()

    # Convert tab chords to DECIBEL format
    nr_of_chords, chord_ids, is_first, is_last = tabchords_to_alignment_arrays(tab_chords, hmm_parameters.alphabet)

    if nr_of_chords < MIN_CHORDS_FOR_ALIGNMENT:
        # Not enough chords to align
        return HMMAlignmentResult(
            alignment_result=AlignmentResult(
                alignments=(),
                tab_chords=tuple(tab_chords),
                timed_chords=(),
                total_distance=0.0,
                normalized_distance=0.0,
            ),
            log_likelihood=float("-inf"),
            transposition=0,
        )

    # Compute emission probabilities
    log_emission = _compute_log_emission_matrix(audio_features.beat_chroma, hmm_parameters)

    # Run Viterbi with transposition search
    viterbi_path, log_likelihood, transposition = _viterbi_with_transposition(
        nr_of_chords,
        chord_ids,
        is_first,
        is_last,
        log_emission,
        hmm_parameters,
        p_f,
        p_b,
    )

    # Convert path to timed chords
    timed_chords = _path_to_timed_chords(viterbi_path, audio_features.beat_times, tab_chords)

    # Create aligned chord pairs
    alignments: list[AlignedChord] = []
    for tc in timed_chords:
        # Find matching tab chord using source_index (preferred) or label fallback
        matching_tab: TabChord | None = None
        idx = tc.source_index
        if idx is not None and 0 <= idx < len(tab_chords):
            matching_tab = tab_chords[idx]
        else:
            # Fallback to label-based search (may fail with duplicate labels)
            matching_tab = next(
                (t for t in tab_chords if t.label == tc.label),
                tab_chords[0] if tab_chords else None,
            )
        if matching_tab is not None:
            alignments.append(
                AlignedChord(
                    tab_chord=matching_tab,
                    timed_chord=tc,
                    distance=0.0,  # HMM doesn't compute per-pair distance
                )
            )

    alignment_result = AlignmentResult(
        alignments=tuple(alignments),
        tab_chords=tuple(tab_chords),
        timed_chords=tuple(timed_chords),
        total_distance=-log_likelihood,  # Convert to distance (lower is better)
        normalized_distance=-log_likelihood / len(timed_chords) if timed_chords else 0.0,
    )

    return HMMAlignmentResult(
        alignment_result=alignment_result,
        log_likelihood=log_likelihood,
        transposition=transposition,
    )


def align_tab_with_audio(
    tab_chords: list[TabChord],
    audio_path: str | Path,
    *,
    hmm_parameters: HMMParameters | None = None,
    p_f: float = 0.05,
    p_b: float = 0.05,
    sampling_rate: int = 22050,
    hop_length: int = 256,
) -> HMMAlignmentResult:
    """Align tab chords with audio file using HMM.

    This is a convenience function that extracts audio features and
    then performs HMM alignment.

    Parameters
    ----------
    tab_chords : list[TabChord]
        Tab chords to align.
    audio_path : str | Path
        Path to the audio file.
    hmm_parameters : HMMParameters | None, optional
        Pre-trained HMM parameters. If None, uses template-based defaults.
    p_f : float, optional
        Forward jump probability for line transitions, by default 0.05.
    p_b : float, optional
        Backward jump probability for line transitions, by default 0.05.
    sampling_rate : int, optional
        Sampling rate for audio loading, by default 22050.
    hop_length : int, optional
        Hop length for feature extraction, by default 256.

    Returns
    -------
    HMMAlignmentResult
        Alignment result with log-likelihood and transposition info.

    Notes
    -----
    Requires librosa to be installed (available via the 'audio' extra).
    """
    audio_features = extract_audio_features(
        audio_path,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
    )
    return align_tab_with_chroma(
        tab_chords,
        audio_features,
        hmm_parameters=hmm_parameters,
        p_f=p_f,
        p_b=p_b,
    )
