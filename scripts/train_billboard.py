#!/usr/bin/env python
"""Train HMM on Billboard 100 dataset with periodic evaluation.

Usage:
    uv run python scripts/train_billboard.py [--test-split 0.2] [--eval-every 10]

This script trains an HMM for chord alignment using the Billboard 100
dataset, evaluating on a held-out test set after every N tracks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Literal

import mir_eval
import numpy as np
from decibel.audio_tab_aligner.hmm_parameters import HMMParameters
from decibel.music_objects.chord import Chord as DECIBELChord
from decibel.music_objects.chord_alphabet import ChordAlphabet
from decibel.music_objects.chord_vocabulary import ChordVocabulary
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chord_parser.decibel import (
    extract_audio_features,
    load_billboard_dataset,
    save_hmm_parameters,
)
from chord_parser.decibel.alignment import _compute_log_emission_matrix


def get_cache_path(audio_path: Path, cache_dir: Path) -> Path:
    """Get cache file path for an audio file."""
    return cache_dir / f"{Path(audio_path).stem}.npz"


def extract_and_cache_single(
    args,
) -> tuple[str, Literal[True], None] | tuple[str, Literal[False], None] | tuple[str, Literal[False], str]:
    """Extract features for a single track and cache to disk."""
    audio_path, cache_dir = args
    audio_path = Path(audio_path)
    cache_path = get_cache_path(audio_path, cache_dir)

    # Skip if already cached
    if cache_path.exists():
        return audio_path.stem, True, None

    try:
        features = extract_audio_features(str(audio_path))
        np.savez(
            cache_path,
            beat_times=features.beat_times,
            beat_chroma=features.beat_chroma,
        )
        return audio_path.stem, False, None
    except Exception as e:
        return audio_path.stem, False, str(e)


def load_cached_features(audio_path: Path, cache_dir: Path) -> tuple[Any, Any]:
    """Load features from cache."""
    cache_path = get_cache_path(audio_path, cache_dir)
    cached = np.load(cache_path)
    return cached["beat_times"], cached["beat_chroma"]


def align_chords_to_beats(
    beat_times: np.ndarray,
    gt_chords: list,
) -> list[str]:
    """Assign ground truth chords to beats by longest overlap."""
    nr_beats = len(beat_times) - 1
    beat_chords: list[str] = []

    for i in range(nr_beats):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]

        longest_duration = 0.0
        longest_chord = "N"

        for tc in gt_chords:
            chord_end = tc.end if tc.end is not None else beat_end + 1

            if tc.start < beat_end and chord_end > beat_start:
                overlap_start = max(tc.start, beat_start)
                overlap_end = min(chord_end, beat_end)
                duration = overlap_end - overlap_start

                if duration > longest_duration:
                    longest_duration = duration
                    longest_chord = "N" if tc.chord is None else tc.chord.to_harte()

        beat_chords.append(longest_chord)

    return beat_chords


def chord_to_alphabet_index(chord_str: str, alphabet: ChordAlphabet) -> int:
    """Convert a Harte chord string to alphabet index."""
    if chord_str == "N":
        return 0

    try:
        chord = DECIBELChord.from_harte_chord_string(chord_str)
        if chord is None:
            return 0
        return int(alphabet.get_index_of_chord_in_alphabet(chord))
    except (KeyError, ValueError, AttributeError):
        return 0


def build_hmm_from_counts(
    alphabet: ChordAlphabet,
    chroma_per_chord: list[list[np.ndarray]],
    trans_counts: np.ndarray,
    init_counts: np.ndarray,
) -> HMMParameters:
    """Build HMM parameters from accumulated counts."""
    alphabet_size = len(alphabet.alphabet_list)

    # Normalize transition matrix
    trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    # Normalize initial distribution
    init = init_counts / init_counts.sum()

    # Compute emission parameters
    obs_mu = np.zeros((alphabet_size, 12))
    obs_sigma = np.zeros((alphabet_size, 12, 12))

    for i in range(alphabet_size):
        if len(chroma_per_chord[i]) > 1:
            chroma_matrix = np.array(chroma_per_chord[i])
            obs_mu[i] = np.mean(chroma_matrix, axis=0)
            obs_sigma[i] = np.cov(chroma_matrix.T, ddof=0)
        else:
            obs_mu[i] = np.ones(12) / 12
            obs_sigma[i] = np.eye(12) * 0.1

    # N state (idx 0) should have low emission probability for typical chroma
    # Use uniform mean with high variance (makes N unlikely for real audio)
    obs_mu[0] = np.ones(12) / 12
    obs_sigma[0] = np.eye(12) * 0.01  # Very tight covariance = unlikely to match

    # Precompute for emission probability
    twelve_log_two_pi = 12 * np.log(2 * np.pi)
    log_det_sigma = np.zeros(alphabet_size)
    sigma_inverse = np.zeros((alphabet_size, 12, 12))

    for i in range(alphabet_size):
        det = np.linalg.det(obs_sigma[i])
        if det > 1e-10:
            log_det_sigma[i] = np.log(det)
            sigma_inverse[i] = np.linalg.pinv(obs_sigma[i])
        else:
            obs_sigma[i] = obs_sigma[i] + np.eye(12) * 0.01
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


def viterbi_decode(hmm: HMMParameters, chroma: np.ndarray) -> list[int]:
    """Run Viterbi decoding to get most likely chord sequence."""
    n_beats = len(chroma)
    n_states = len(hmm.alphabet.alphabet_list)

    # Note: _compute_log_emission_matrix returns (n_states, n_beats), need to transpose
    log_emission = _compute_log_emission_matrix(chroma, hmm).T  # Now (n_beats, n_states)
    log_trans = np.log(hmm.trans + 1e-10)
    log_init = np.log(hmm.init + 1e-10)

    viterbi = np.full((n_beats, n_states), -np.inf)
    backptr = np.zeros((n_beats, n_states), dtype=int)

    viterbi[0] = log_init + log_emission[0]

    for t in range(1, n_beats):
        for j in range(n_states):
            scores = viterbi[t - 1] + log_trans[:, j]
            backptr[t, j] = np.argmax(scores)
            viterbi[t, j] = scores[backptr[t, j]] + log_emission[t, j]

    path = [int(np.argmax(viterbi[-1]))]
    for t in range(n_beats - 1, 0, -1):
        path.append(backptr[t, path[-1]])
    path.reverse()

    return path


def chord_index_to_label(idx: int, alphabet: ChordAlphabet) -> str:
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


def compute_chord_metrics(
    gt_intervals: np.ndarray,
    gt_labels: list[str],
    pred_intervals: np.ndarray,
    pred_labels: list[str],
) -> dict[str, float]:
    """Compute chord evaluation metrics using mir_eval utilities.

    Uses merge_labeled_intervals to align predictions with ground truth,
    avoiding the overlap bugs in mir_eval.chord.evaluate().
    """
    # Adjust intervals to same time range
    t_max = max(gt_intervals[-1, 1], pred_intervals[-1, 1])
    gt_adj, gt_labels_adj = mir_eval.util.adjust_intervals(
        gt_intervals, gt_labels, t_min=0.0, t_max=t_max
    )
    pred_adj, pred_labels_adj = mir_eval.util.adjust_intervals(
        pred_intervals, pred_labels, t_min=0.0, t_max=t_max,
        start_label="N", end_label="N"
    )

    # Merge into common intervals
    merged, merged_gt, merged_pred = mir_eval.util.merge_labeled_intervals(
        gt_adj, gt_labels_adj, pred_adj, pred_labels_adj
    )
    durations = mir_eval.util.intervals_to_durations(merged)
    total_duration = np.sum(durations)

    # Encode chords for comparison
    ref_roots = []
    est_roots = []
    ref_thirds = []
    est_thirds = []
    ref_majmin = []
    est_majmin = []

    for gl, pl in zip(merged_gt, merged_pred):
        try:
            ref_enc = mir_eval.chord.encode(gl)
            ref_roots.append(ref_enc[0])  # root pitch class
            ref_thirds.append((ref_enc[0], ref_enc[1][3] if len(ref_enc[1]) > 3 else 0))  # root + third
            # majmin: 0=N, 1=maj, 2=min
            if gl == "N":
                ref_majmin.append(0)
            elif ":min" in gl or gl.endswith("m"):
                ref_majmin.append(2)
            else:
                ref_majmin.append(1)
        except Exception:
            ref_roots.append(-1)
            ref_thirds.append((-1, -1))
            ref_majmin.append(-1)

        try:
            est_enc = mir_eval.chord.encode(pl)
            est_roots.append(est_enc[0])
            est_thirds.append((est_enc[0], est_enc[1][3] if len(est_enc[1]) > 3 else 0))
            if pl == "N":
                est_majmin.append(0)
            elif ":min" in pl or pl.endswith("m"):
                est_majmin.append(2)
            else:
                est_majmin.append(1)
        except Exception:
            est_roots.append(-1)
            est_thirds.append((-1, -1))
            est_majmin.append(-1)

    # Compute weighted accuracies
    root_match = np.array([
        r == e if r >= 0 and e >= 0 else False
        for r, e in zip(ref_roots, est_roots)
    ])
    thirds_match = np.array([
        r == e if r[0] >= 0 and e[0] >= 0 else False
        for r, e in zip(ref_thirds, est_thirds)
    ])
    majmin_match = np.array([
        r == e if r >= 0 and e >= 0 else False
        for r, e in zip(ref_majmin, est_majmin)
    ])

    return {
        "Root": float(np.sum(root_match * durations) / total_duration),
        "Thirds": float(np.sum(thirds_match * durations) / total_duration),
        "Majmin": float(np.sum(majmin_match * durations) / total_duration),
    }


def evaluate_on_test_set(
    hmm: HMMParameters,
    test_data: list,
    cache_dir: Path,
) -> dict[str, float]:
    """Evaluate HMM on test set, loading features from cache."""
    all_scores: dict[str, list[float]] = {}

    for audio_path, gt_chords in test_data:
        try:
            # Load features from cache
            beat_times, beat_chroma = load_cached_features(audio_path, cache_dir)

            pred_indices = viterbi_decode(hmm, beat_chroma)

            n_beats = len(pred_indices)

            pred_intervals = []
            pred_labels = []
            for i in range(n_beats):
                if i < len(beat_times) - 1:
                    pred_intervals.append([beat_times[i], beat_times[i + 1]])
                    pred_labels.append(chord_index_to_label(pred_indices[i], hmm.alphabet))

            pred_intervals = np.array(pred_intervals)
            gt_intervals = np.array([[tc.start, tc.end] for tc in gt_chords])
            gt_labels_list = [tc.label for tc in gt_chords]

            scores = compute_chord_metrics(
                gt_intervals, gt_labels_list, pred_intervals, pred_labels
            )

            for metric, value in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(value)

        except Exception as e:
            tqdm.write(f"  EVAL ERROR on {Path(audio_path).stem}: {e}")
            continue

    return {metric: float(np.mean(values)) for metric, values in all_scores.items()}


def main() -> None:
    """Train HMM on Billboard dataset with periodic evaluation."""
    parser = argparse.ArgumentParser(description="Train HMM on Billboard dataset")
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate on test set after every N tracks (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for feature extraction (default: CPU count)",
    )
    args = parser.parse_args()

    dataset_dir = project_root / "billboard_100"
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("Loading Billboard 100 dataset...")
    all_data = load_billboard_dataset(dataset_dir)
    print(f"Loaded {len(all_data)} tracks")

    if len(all_data) == 0:
        print("No training data found!")
        sys.exit(1)

    # Split into train/test
    random.seed(args.seed)
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    n_test = int(len(all_data) * args.test_split)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    training_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]

    print(f"Train set: {len(training_data)} tracks")
    print(f"Test set: {len(test_data)} tracks")

    # Save test set info
    test_audio_paths = [str(audio_path) for audio_path, _ in test_data]
    test_manifest_path = checkpoint_dir / "test_audio_paths.json"
    with open(test_manifest_path, "w") as f:
        json.dump(test_audio_paths, f, indent=2)

    # Initialize training state
    vocabulary = ChordVocabulary.generate_chroma_major_minor()
    alphabet = ChordAlphabet(vocabulary)
    alphabet_size = len(alphabet.alphabet_list)

    chroma_per_chord: list[list[np.ndarray]] = [[] for _ in range(alphabet_size)]
    trans_counts = np.ones((alphabet_size, alphabet_size))  # Laplace smoothing
    init_counts = np.ones(alphabet_size)

    # Feature cache directory
    cache_dir = project_root / "cache" / "billboard_features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio paths that need caching
    all_audio_paths = [audio_path for audio_path, _ in training_data + test_data]
    work_items = [(audio_path, cache_dir) for audio_path in all_audio_paths]

    # Check how many are already cached
    n_already_cached = sum(1 for audio_path in all_audio_paths if get_cache_path(audio_path, cache_dir).exists())
    n_to_extract = len(all_audio_paths) - n_already_cached

    print(f"\nFeature caching ({cache_dir})")
    print(f"  Total: {len(all_audio_paths)}, Cached: {n_already_cached}, To extract: {n_to_extract}")

    if n_to_extract > 0:
        n_workers = args.workers or os.cpu_count() or 4
        print(f"  Extracting with {n_workers} workers...")

        with mp.Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(extract_and_cache_single, work_items),
                    total=len(work_items),
                    desc="  Caching features",
                    unit="track",
                )
            )

        # Report errors
        errors = [(name, err) for name, cached, err in results if err]
        if errors:
            print(f"  Errors: {len(errors)}")
            for name, err in errors[:5]:
                print(f"    {name}: {err}")
    else:
        print("  All features already cached.")

    print()
    print(f"Training with evaluation every {args.eval_every} tracks...")
    print("=" * 70)

    start_time = time.time()

    # Training loop - load features one at a time from cache
    for track_num, (audio_path, gt_chords) in enumerate(tqdm(training_data, desc="Training", unit="track"), 1):
        try:
            # Load features from cache
            beat_times, beat_chroma = load_cached_features(audio_path, cache_dir)

            # Align GT chords to detected beats
            beat_chords = align_chords_to_beats(beat_times, gt_chords)

            # Convert chord strings to alphabet indices
            chord_indices = [chord_to_alphabet_index(c, alphabet) for c in beat_chords]

            # Skip leading N (silence/intro) to avoid biasing toward N
            first_non_n = 0
            for i, idx in enumerate(chord_indices):
                if idx != 0:  # 0 is N
                    first_non_n = i
                    break

            # Collect chroma per chord (skip N segments to reduce N bias)
            for i, chord_idx in enumerate(chord_indices):
                if i < len(beat_chroma) and chord_idx != 0:  # Skip N
                    chroma_per_chord[chord_idx].append(beat_chroma[i])

            # Count initial chord (first non-N chord)
            if first_non_n < len(chord_indices):
                init_counts[chord_indices[first_non_n]] += 1

            # Count transitions (skip N->X and X->N transitions)
            for i in range(len(chord_indices) - 1):
                from_idx = chord_indices[i]
                to_idx = chord_indices[i + 1]
                if from_idx != 0 and to_idx != 0:  # Skip N transitions
                    trans_counts[from_idx, to_idx] += 1
                elif from_idx != 0:  # Allow X->N but don't count
                    pass
                elif to_idx != 0:  # Allow N->X with reduced weight
                    trans_counts[from_idx, to_idx] += 0.1

        except Exception as e:
            tqdm.write(f"  ERROR on {Path(audio_path).stem}: {e}")
            continue

        # Evaluate every N tracks
        if track_num % args.eval_every == 0:
            hmm = build_hmm_from_counts(alphabet, chroma_per_chord, trans_counts, init_counts)
            metrics = evaluate_on_test_set(hmm, test_data, cache_dir)
            tqdm.write(
                f"  [{track_num}] Root: {metrics.get('Root', 0):.4f}  "
                f"Majmin: {metrics.get('Majmin', 0):.4f}  "
                f"Thirds: {metrics.get('Thirds', 0):.4f}  "
                f"Triads: {metrics.get('Triads', 0):.4f}"
            )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {elapsed:.1f} seconds")

    # Build final HMM and save
    trained_hmm = build_hmm_from_counts(alphabet, chroma_per_chord, trans_counts, init_counts)

    # Final evaluation
    print("\nFinal evaluation on test set:")
    final_metrics = evaluate_on_test_set(trained_hmm, test_data, cache_dir)
    for metric in sorted(final_metrics.keys()):
        print(f"  {metric:20s}: {final_metrics[metric]:.4f}")

    # Save checkpoint
    checkpoint_path = checkpoint_dir / "billboard_hmm.npz"
    save_hmm_parameters(trained_hmm, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method("spawn", force=True)
    main()
