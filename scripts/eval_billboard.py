#!/usr/bin/env python
"""Evaluate trained HMM on Billboard test set using mir_eval.

Usage:
    uv run python scripts/eval_billboard.py

This script evaluates a trained HMM checkpoint on the held-out test set
using mir_eval chord metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mir_eval
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chord_parser.decibel import (
    extract_audio_features,
    load_billboard_chords,
    load_hmm_parameters,
)
from chord_parser.decibel.alignment import _compute_log_emission_matrix


def viterbi_decode(
    hmm,
    chroma: np.ndarray,
) -> list[int]:
    """Run Viterbi decoding to get most likely chord sequence.

    Parameters
    ----------
    hmm : HMMParameters
        Trained HMM parameters.
    chroma : np.ndarray
        Beat-synchronous chroma features, shape (n_beats, 12).

    Returns
    -------
    list[int]
        Predicted chord indices for each beat.
    """
    n_beats = len(chroma)
    n_states = len(hmm.alphabet.alphabet_list)

    # Compute emission probabilities
    # Note: _compute_log_emission_matrix returns (n_states, n_beats), need to transpose
    log_emission = _compute_log_emission_matrix(chroma, hmm).T  # Now (n_beats, n_states)

    # Initialize
    log_trans = np.log(hmm.trans + 1e-10)
    log_init = np.log(hmm.init + 1e-10)

    # Viterbi
    viterbi = np.full((n_beats, n_states), -np.inf)
    backptr = np.zeros((n_beats, n_states), dtype=int)

    viterbi[0] = log_init + log_emission[0]

    for t in range(1, n_beats):
        for j in range(n_states):
            scores = viterbi[t - 1] + log_trans[:, j]
            backptr[t, j] = np.argmax(scores)
            viterbi[t, j] = scores[backptr[t, j]] + log_emission[t, j]

    # Backtrace
    path = [int(np.argmax(viterbi[-1]))]
    for t in range(n_beats - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    return path


def chord_index_to_label(idx: int, alphabet) -> str:
    """Convert chord index to mir_eval compatible Harte label."""
    label = alphabet.alphabet_list[idx]
    if label is None or label == "N":
        return "N"

    # Convert DECIBEL format (C, Cm) to Harte format (C:maj, C:min)
    label = str(label)
    if label.endswith("m"):
        # Minor chord: "Cm" -> "C:min", "C#m" -> "C#:min"
        root = label[:-1]
        return f"{root}:min"
    else:
        # Major chord: "C" -> "C:maj"
        return f"{label}:maj"


def evaluate_track(
    audio_path: Path,
    gt_chords: list,
    hmm,
) -> dict:
    """Evaluate HMM predictions on a single track.

    Returns
    -------
    dict
        Dictionary with mir_eval chord metrics.
    """
    # Extract features
    features = extract_audio_features(str(audio_path))

    # Run Viterbi
    pred_indices = viterbi_decode(hmm, features.beat_chroma)

    # Convert predictions to intervals and labels
    beat_times = features.beat_times
    n_beats = len(pred_indices)

    pred_intervals = []
    pred_labels = []
    for i in range(n_beats):
        if i < len(beat_times) - 1:
            pred_intervals.append([beat_times[i], beat_times[i + 1]])
            pred_labels.append(chord_index_to_label(pred_indices[i], hmm.alphabet))

    pred_intervals = np.array(pred_intervals)
    pred_labels = np.array(pred_labels)

    # Convert ground truth to intervals and labels
    gt_intervals = []
    gt_labels = []
    for tc in gt_chords:
        gt_intervals.append([tc.start, tc.end])
        gt_labels.append(tc.label)

    gt_intervals = np.array(gt_intervals)
    gt_labels = np.array(gt_labels)

    # Compute mir_eval metrics
    # Use overlap-based comparison
    scores = mir_eval.chord.evaluate(gt_intervals, gt_labels, pred_intervals, pred_labels)

    return scores


def main() -> None:
    """Evaluate HMM on test set."""
    checkpoint_dir = project_root / "checkpoints"
    dataset_dir = project_root / "billboard_100"

    # Load checkpoint
    checkpoint_path = checkpoint_dir / "billboard_hmm.npz"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run train_billboard.py first.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    hmm = load_hmm_parameters(checkpoint_path)

    # Load test set paths
    test_paths_file = checkpoint_dir / "test_audio_paths.json"
    if not test_paths_file.exists():
        print(f"Test paths not found: {test_paths_file}")
        sys.exit(1)

    with open(test_paths_file) as f:
        test_audio_paths = json.load(f)

    print(f"Evaluating on {len(test_audio_paths)} test tracks...")

    # Collect scores
    all_scores: dict[str, list[float]] = {}

    for i, audio_path in enumerate(test_audio_paths):
        audio_path = Path(audio_path)
        track_id = audio_path.stem

        # Load ground truth
        chord_path = dataset_dir / "chords" / f"{track_id}.json"
        gt_chords = load_billboard_chords(chord_path)

        print(f"  [{i+1}/{len(test_audio_paths)}] {track_id}...", end=" ", flush=True)

        try:
            scores = evaluate_track(audio_path, gt_chords, hmm)

            for metric, value in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(value)

            # Print key metric for this track
            print(f"root={scores['Root']:.3f}, majmin={scores['Majmin']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for metric in sorted(all_scores.keys()):
        values = all_scores[metric]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric:20s}: {mean:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()
