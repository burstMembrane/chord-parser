#!/usr/bin/env python
"""Demo script for aligning tab chords with Chordino annotations.

Run with: uv run python examples/align_upside_down.py
"""

from pathlib import Path

from chord_parser import tab_parser
from chord_parser.alignment import (
    align_chords,
    chord_distance_exact,
    chord_distance_flexible,
    chord_distance_pitchclass,
    extract_tab_chords,
    load_chordino_json,
)


def main() -> None:
    """Run the alignment demo."""
    # Paths
    project_root = Path(__file__).parent.parent
    chordino_path = project_root / "upsidedown" / "1285_chordino.json"
    tab_path = project_root / "testdata" / "diana-ross-upside-down.md"

    if not chordino_path.exists():
        print(f"Error: Chordino file not found: {chordino_path}")
        return
    if not tab_path.exists():
        print(f"Error: Tab file not found: {tab_path}")
        return

    # Load Chordino annotations
    print(f"Loading Chordino annotations from: {chordino_path}")
    timed_chords = load_chordino_json(chordino_path)
    print(f"  Found {len(timed_chords)} timed chords")

    # Parse tab sheet
    print(f"\nParsing tab from: {tab_path}")
    content = tab_path.read_text()

    # Extract tab content from markdown (between ``` markers)
    lines = content.split("\n")
    in_code_block = False
    tab_lines = []
    for line in lines:
        if line.strip() == "```":
            in_code_block = not in_code_block
            continue
        if in_code_block:
            tab_lines.append(line)

    tab_content = "\n".join(tab_lines)
    sheet = tab_parser.parse(tab_content)
    tab_chords = extract_tab_chords(sheet)
    print(f"  Found {len(tab_chords)} tab chords across {len(sheet.sections)} sections")

    # Show sections
    print("\n  Sections:")
    for section in sheet.sections:
        section_chords = [c for c in tab_chords if c.section == section.name]
        print(f"    - {section.name}: {len(section_chords)} chords")

    # Run DTW alignment with all three distance functions
    print("\nRunning DTW alignment...")

    # Exact matching
    result_exact = align_chords(tab_chords, timed_chords, distance_fn=chord_distance_exact)

    # Flexible (heuristic) matching
    result_flex = align_chords(tab_chords, timed_chords, distance_fn=chord_distance_flexible)

    # Pitch-class (Jaccard) matching
    result_pitch = align_chords(tab_chords, timed_chords, distance_fn=chord_distance_pitchclass)

    # Summary comparison
    def summarize(result, name):
        """Summarize alignment results."""
        perfect = sum(1 for a in result.alignments if a.distance == 0.0)
        close = sum(1 for a in result.alignments if a.distance <= 0.25)
        near = sum(1 for a in result.alignments if a.distance <= 0.4)
        n = len(result.alignments)
        print(f"  {name}:")
        print(f"    Perfect (dist=0):  {perfect:3d} ({100*perfect/n:5.1f}%)")
        print(f"    Close (dist<=0.25):{close:3d} ({100*close/n:5.1f}%)")
        print(f"    Near (dist<=0.4):  {near:3d} ({100*near/n:5.1f}%)")
        print(f"    Normalized dist:   {result.normalized_distance:.4f}")

    print(f"\n{'='*60}")
    print("ALIGNMENT COMPARISON")
    print(f"{'='*60}")
    print(f"  Tab chords:   {len(tab_chords)}")
    print(f"  Timed chords: {len(timed_chords)}")
    print()
    summarize(result_exact, "EXACT")
    print()
    summarize(result_flex, "FLEXIBLE (heuristic)")
    print()
    summarize(result_pitch, "PITCH-CLASS (Jaccard)")

    result = result_pitch  # Use pitch-class for detailed output

    # Show first N alignments
    n_show = 30
    print(f"\n{'='*60}")
    print(f"FIRST {n_show} ALIGNMENTS")
    print(f"{'='*60}")
    print(f"{'Tab':12} {'Chordino':12} {'Start':>8} {'End':>8} {'Dist':>5} Section")
    print("-" * 70)
    for a in result.alignments[:n_show]:
        end = f"{a.timed_chord.end:.2f}" if a.timed_chord.end else "None"
        match = "*" if a.distance == 0.0 else " "
        print(
            f"{a.tab_chord.label:12} {a.timed_chord.label:12} "
            f"{a.timed_chord.start:8.2f} {end:>8} {a.distance:5.1f}{match} {a.tab_chord.section}"
        )

    # Show mismatches
    mismatches = [a for a in result.alignments if a.distance > 0.0]
    if mismatches:
        print(f"\n{'='*60}")
        print(f"MISMATCHES ({len(mismatches)} total)")
        print(f"{'='*60}")
        print(f"{'Tab':12} {'Chordino':12} {'Start':>8} Section")
        print("-" * 50)
        for a in mismatches[:20]:
            print(f"{a.tab_chord.label:12} {a.timed_chord.label:12} {a.timed_chord.start:8.2f} {a.tab_chord.section}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")


if __name__ == "__main__":
    main()
