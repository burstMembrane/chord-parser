#!/usr/bin/env python
"""Compare different chord detection algorithms.

This script compares chord detection results from multiple algorithms
(Chordino, Essentia, madmom) against a tab sheet's chord progression.

Run with:
    uv run --group audio python examples/compare_chord_detectors.py

Or install audio dependencies first:
    uv sync --group audio
    uv run python examples/compare_chord_detectors.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from chord_parser import tab_parser
from chord_parser.alignment import (
    align_chords,
    chord_distance_flexible,
    chord_distance_pitchclass,
    extract_tab_chords,
)
from chord_parser.alignment.models import TimedChord


@dataclass
class DetectorResult:
    """Results from a chord detector."""

    name: str
    timed_chords: list[TimedChord]
    alignment_distance: float
    normalized_distance: float
    perfect_matches: int
    close_matches: int
    mismatches: int
    total_chords: int

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as perfect + close matches / total."""
        if self.total_chords == 0:
            return 0.0
        return (self.perfect_matches + self.close_matches) / self.total_chords


def load_tab_chords(tab_path: Path) -> list:
    """Load and parse chords from a tab file.

    Handles markdown files with code blocks containing tab content.
    """
    content = tab_path.read_text()

    # Extract content from code blocks if present
    lines = content.split("\n")
    in_code_block = False
    tab_lines = []
    for line in lines:
        if line.strip() == "```":
            in_code_block = not in_code_block
            continue
        if in_code_block:
            tab_lines.append(line)

    # If no code blocks found, use all content
    if not tab_lines:
        tab_content = content
    else:
        tab_content = "\n".join(tab_lines)

    sheet = tab_parser.parse(tab_content)
    return extract_tab_chords(sheet)


def run_essentia(audio_path: Path, verbose: bool = False) -> list[TimedChord]:
    """Run Essentia chord detection."""
    try:
        from chord_parser.alignment.essentia_extractor import extract_chords_essentia

        if verbose:
            print("  Running Essentia ChordsDetection...")
        return extract_chords_essentia(audio_path)
    except ImportError as e:
        print(f"  Essentia not available: {e}")
        print("  Install with: uv sync --group audio")
        return []


def run_madmom(audio_path: Path, verbose: bool = False) -> list[TimedChord]:
    """Run madmom chord detection."""
    try:
        from chord_parser.alignment.madmom_extractor import extract_chords_madmom

        if verbose:
            print("  Running madmom DeepChroma...")
        return extract_chords_madmom(audio_path)
    except ImportError as e:
        print(f"  madmom not available: {e}")
        print("  Install with: uv sync --group audio")
        return []


def evaluate_detector(
    name: str,
    timed_chords: list[TimedChord],
    tab_chords: list,
    threshold: float = 0.25,
) -> DetectorResult:
    """Evaluate a detector's results against tab chords."""
    if not timed_chords:
        return DetectorResult(
            name=name,
            timed_chords=[],
            alignment_distance=float("inf"),
            normalized_distance=float("inf"),
            perfect_matches=0,
            close_matches=0,
            mismatches=0,
            total_chords=0,
        )

    # Run DTW alignment
    result = align_chords(
        tab_chords,
        timed_chords,
        distance_fn=chord_distance_pitchclass,
    )

    # Count match quality
    perfect = sum(1 for a in result.alignments if a.distance == 0.0)
    close = sum(1 for a in result.alignments if 0 < a.distance <= threshold)
    mismatch = sum(1 for a in result.alignments if a.distance > threshold)

    return DetectorResult(
        name=name,
        timed_chords=timed_chords,
        alignment_distance=result.total_distance,
        normalized_distance=result.normalized_distance,
        perfect_matches=perfect,
        close_matches=close,
        mismatches=mismatch,
        total_chords=len(result.alignments),
    )


def print_results(results: list[DetectorResult], verbose: bool = False) -> None:
    """Print comparison results."""
    print("\n" + "=" * 60)
    print("CHORD DETECTION COMPARISON RESULTS")
    print("=" * 60)

    # Sort by accuracy (descending)
    sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)

    for i, r in enumerate(sorted_results, 1):
        print(f"\n{i}. {r.name}")
        print("-" * 40)

        if r.total_chords == 0:
            print("   (No results - detector may not be available)")
            continue

        print(f"   Detected chords: {len(r.timed_chords)}")
        print(f"   Aligned chords:  {r.total_chords}")
        print(f"   Perfect matches: {r.perfect_matches} ({100*r.perfect_matches/r.total_chords:.1f}%)")
        print(f"   Close matches:   {r.close_matches} ({100*r.close_matches/r.total_chords:.1f}%)")
        print(f"   Mismatches:      {r.mismatches} ({100*r.mismatches/r.total_chords:.1f}%)")
        print(f"   Accuracy:        {r.accuracy*100:.1f}%")
        print(f"   DTW distance:    {r.alignment_distance:.2f} (normalized: {r.normalized_distance:.3f})")

        if verbose and r.timed_chords:
            print("\n   First 10 detected chords:")
            for tc in r.timed_chords[:10]:
                print(f"     {tc.start:6.2f}s - {tc.end or 0:6.2f}s: {tc.label}")


def export_results(
    results: list[DetectorResult],
    output_path: Path,
) -> None:
    """Export results to JSON."""
    data = {
        "results": [
            {
                "name": r.name,
                "detected_chords": len(r.timed_chords),
                "aligned_chords": r.total_chords,
                "perfect_matches": r.perfect_matches,
                "close_matches": r.close_matches,
                "mismatches": r.mismatches,
                "accuracy": r.accuracy,
                "dtw_distance": r.alignment_distance,
                "normalized_distance": r.normalized_distance,
                "chords": [
                    {
                        "start": tc.start,
                        "end": tc.end,
                        "label": tc.label,
                    }
                    for tc in r.timed_chords
                ],
            }
            for r in results
        ]
    }

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults exported to: {output_path}")


def main() -> None:
    """Run chord detection comparison."""
    parser = argparse.ArgumentParser(
        description="Compare chord detection algorithms against a tab sheet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to audio file (WAV, MP3, etc.)",
    )
    parser.add_argument(
        "tab",
        type=Path,
        help="Path to tab file (markdown with chord notation)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including detected chords",
    )
    parser.add_argument(
        "--essentia-only",
        action="store_true",
        help="Only run Essentia detector",
    )
    parser.add_argument(
        "--madmom-only",
        action="store_true",
        help="Only run madmom detector",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}")
        return

    if not args.tab.exists():
        print(f"Error: Tab file not found: {args.tab}")
        return

    print(f"Audio: {args.audio}")
    print(f"Tab:   {args.tab}")

    # Load tab chords
    print("\nLoading tab chords...")
    tab_chords = load_tab_chords(args.tab)
    print(f"  Found {len(tab_chords)} chords in tab")

    # Determine which detectors to run
    run_all = not args.essentia_only and not args.madmom_only

    results: list[DetectorResult] = []

    # Run Essentia
    if run_all or args.essentia_only:
        print("\nRunning Essentia...")
        essentia_chords = run_essentia(args.audio, verbose=args.verbose)
        essentia_result = evaluate_detector("Essentia (HPCP template)", essentia_chords, tab_chords)
        results.append(essentia_result)

    # Run madmom
    if run_all or args.madmom_only:
        print("\nRunning madmom...")
        madmom_chords = run_madmom(args.audio, verbose=args.verbose)
        madmom_result = evaluate_detector("madmom (DeepChroma + CRF)", madmom_chords, tab_chords)
        results.append(madmom_result)

    # Print results
    print_results(results, verbose=args.verbose)

    # Export if requested
    if args.output:
        export_results(results, args.output)


if __name__ == "__main__":
    main()
