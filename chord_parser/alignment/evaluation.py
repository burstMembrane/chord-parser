"""Chord alignment evaluation using mir_eval.

This module provides evaluation metrics for chord alignment results,
following the MIREX chord estimation evaluation methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chord_parser.alignment.models import AlignmentResult, TimedChord


@dataclass
class EvaluationResult:
    """Results from evaluating alignment against ground truth.

    Attributes
    ----------
    csr : float
        Chord Symbol Recall (time-weighted accuracy for majmin comparison).
    root_accuracy : float
        Fraction of time where root notes match.
    thirds_accuracy : float
        Fraction of time where thirds (major/minor) match.
    coverage : float
        Fraction of audio duration covered by aligned chords.
    """

    csr: float
    root_accuracy: float
    thirds_accuracy: float
    coverage: float


def _chord_to_mir_eval_label(chord_label: str) -> str:
    """Convert a chord label to mir_eval compatible format.

    Parameters
    ----------
    chord_label : str
        Chord label in pychord or Harte format.

    Returns
    -------
    str
        Chord label in mir_eval format.
    """
    # Handle "N" for no chord
    if chord_label == "N" or not chord_label:
        return "N"

    # mir_eval expects Harte-style labels without absolute bass notes
    # e.g., "G:min7" is ok, but "G:min7/D" should be "G:min7/5"
    # For simplicity, we strip bass notes since mir_eval's majmin comparison
    # ignores them anyway
    try:
        from chord_parser.converter import from_pychord, from_harte

        if ":" in chord_label:
            # Already Harte format
            chord = from_harte(chord_label)
        else:
            # pychord format
            chord = from_pychord(chord_label)

        # Return just root:quality without bass (mir_eval majmin ignores bass anyway)
        return f"{chord.root}:{chord.quality}"
    except (ValueError, ImportError):
        # If conversion fails, try to at least strip the bass
        if "/" in chord_label:
            return chord_label.split("/")[0]
        return chord_label


def evaluate_against_ground_truth(
    alignment_result: AlignmentResult,
    ground_truth: list[TimedChord],
    audio_duration: float | None = None,
) -> EvaluationResult:
    """Evaluate alignment result against ground truth annotations.

    Uses mir_eval's chord evaluation methodology to compute metrics.

    Parameters
    ----------
    alignment_result : AlignmentResult
        The alignment result to evaluate.
    ground_truth : list[TimedChord]
        Ground truth chord annotations with timing.
    audio_duration : float | None
        Total audio duration. If None, uses last ground truth end time.

    Returns
    -------
    EvaluationResult
        Evaluation metrics.
    """
    try:
        import mir_eval
    except ImportError:
        msg = "mir_eval is required for evaluation. Install with: pip install mir_eval"
        raise ImportError(msg) from None

    if not ground_truth:
        return EvaluationResult(csr=0.0, root_accuracy=0.0, thirds_accuracy=0.0, coverage=0.0)

    # Determine audio duration
    if audio_duration is None:
        audio_duration = max(tc.end or tc.start for tc in ground_truth)

    # Build ground truth intervals and labels
    ref_intervals_list: list[list[float]] = []
    ref_labels: list[str] = []
    for tc in ground_truth:
        start = tc.start
        end = tc.end if tc.end is not None else audio_duration
        ref_intervals_list.append([start, end])
        ref_labels.append(_chord_to_mir_eval_label(tc.label))

    ref_intervals: NDArray[Any] = np.array(ref_intervals_list)

    # Build estimated intervals and labels from alignment
    if not alignment_result.alignments:
        # No alignments - return zeros
        return EvaluationResult(csr=0.0, root_accuracy=0.0, thirds_accuracy=0.0, coverage=0.0)

    est_intervals_list: list[list[float]] = []
    est_labels: list[str] = []

    # Create intervals from alignments
    # Each aligned chord gets timing from its timed_chord
    for aligned in alignment_result.alignments:
        tc = aligned.timed_chord
        start = tc.start
        end = tc.end if tc.end is not None else start + 1.0  # Default 1 second duration
        est_intervals_list.append([start, end])
        est_labels.append(_chord_to_mir_eval_label(tc.label))

    est_intervals: NDArray[Any] = np.array(est_intervals_list)

    # Adjust intervals to match reference bounds
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels,
        ref_intervals.min(), ref_intervals.max(),
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD
    )

    # Merge intervals for comparison
    intervals, ref_merged, est_merged = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    durations = mir_eval.util.intervals_to_durations(intervals)

    # Compute metrics
    # Major/minor comparison (most common metric)
    comparisons_majmin = mir_eval.chord.majmin(ref_merged, est_merged)
    csr = mir_eval.chord.weighted_accuracy(comparisons_majmin, durations)

    # Root comparison
    comparisons_root = mir_eval.chord.root(ref_merged, est_merged)
    root_accuracy = mir_eval.chord.weighted_accuracy(comparisons_root, durations)

    # Thirds comparison
    comparisons_thirds = mir_eval.chord.thirds(ref_merged, est_merged)
    thirds_accuracy = mir_eval.chord.weighted_accuracy(comparisons_thirds, durations)

    # Coverage: fraction of audio with aligned chords
    aligned_duration = sum(
        (a.timed_chord.end or a.timed_chord.start + 1.0) - a.timed_chord.start
        for a in alignment_result.alignments
    )
    coverage = min(aligned_duration / audio_duration, 1.0) if audio_duration > 0 else 0.0

    return EvaluationResult(
        csr=csr,
        root_accuracy=root_accuracy,
        thirds_accuracy=thirds_accuracy,
        coverage=coverage,
    )


def compare_chord_sequences(
    sequence1: list[TimedChord],
    sequence2: list[TimedChord],
) -> dict[str, float]:
    """Compare two timed chord sequences using mir_eval.

    Useful for comparing different chord estimation methods or
    ground truth against predictions.

    Parameters
    ----------
    sequence1 : list[TimedChord]
        First chord sequence (typically reference/ground truth).
    sequence2 : list[TimedChord]
        Second chord sequence (typically estimation).

    Returns
    -------
    dict[str, float]
        Dictionary with CSR, root, thirds, and segmentation metrics.
    """
    try:
        import mir_eval
    except ImportError:
        msg = "mir_eval is required. Install with: pip install mir_eval"
        raise ImportError(msg) from None

    if not sequence1 or not sequence2:
        return {"csr": 0.0, "root": 0.0, "thirds": 0.0}

    # Build intervals and labels for sequence 1
    ref_intervals_list: list[list[float]] = []
    ref_labels: list[str] = []
    for tc in sequence1:
        end = tc.end if tc.end is not None else tc.start + 1.0
        ref_intervals_list.append([tc.start, end])
        ref_labels.append(_chord_to_mir_eval_label(tc.label))
    ref_intervals: NDArray[Any] = np.array(ref_intervals_list)

    # Build intervals and labels for sequence 2
    est_intervals_list: list[list[float]] = []
    est_labels: list[str] = []
    for tc in sequence2:
        end = tc.end if tc.end is not None else tc.start + 1.0
        est_intervals_list.append([tc.start, end])
        est_labels.append(_chord_to_mir_eval_label(tc.label))
    est_intervals: NDArray[Any] = np.array(est_intervals_list)

    # Adjust and merge
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels,
        ref_intervals.min(), ref_intervals.max(),
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD
    )

    intervals, ref_merged, est_merged = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )
    durations = mir_eval.util.intervals_to_durations(intervals)

    # Compute metrics
    csr = mir_eval.chord.weighted_accuracy(
        mir_eval.chord.majmin(ref_merged, est_merged), durations
    )
    root = mir_eval.chord.weighted_accuracy(
        mir_eval.chord.root(ref_merged, est_merged), durations
    )
    thirds = mir_eval.chord.weighted_accuracy(
        mir_eval.chord.thirds(ref_merged, est_merged), durations
    )

    return {"csr": csr, "root": root, "thirds": thirds}
