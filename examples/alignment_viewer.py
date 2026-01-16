#!/usr/bin/env python
"""Streamlit app for viewing chord alignment results.

Run with: uv run streamlit run examples/alignment_viewer.py

Features:
- Alignment between tab chords and audio chord predictions
- Comparison against ground truth annotations
- Pitch class similarity analysis
- Auto-transposition detection
- Visual chord timeline
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

from chord_parser import tab_parser
from chord_parser.alignment import (
    align_chords,
    align_with_transposition,
    chord_similarity,
    compute_alignment_metrics,
    extract_tab_chords,
)
from chord_parser.alignment.models import TabChord, TimedChord
from chord_parser.converter import from_harte, from_pychord
from chord_parser.models import Chord
from chord_parser.pitch_class import (
    chord_pitch_similarity,
    chord_to_pitch_classes,
    quality_category,
    roots_match,
)

# Color scheme for match quality
MATCH_COLORS = {
    "exact": "#22c55e",  # green
    "root_only": "#eab308",  # yellow
    "mismatch": "#ef4444",  # red
    "unaligned": "#6b7280",  # gray
}

# Section colors for visual distinction
SECTION_COLORS = [
    "#6366f1",  # indigo
    "#ec4899",  # pink
    "#0ea5e9",  # sky
    "#a855f7",  # purple
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#84cc16",  # lime
    "#f43f5e",  # rose
]


def get_section_color(section: str, sections: list[str]) -> str:
    """Get color for a section."""
    if section in sections:
        idx = sections.index(section)
        return SECTION_COLORS[idx % len(SECTION_COLORS)]
    return SECTION_COLORS[0]


def load_ground_truth(path: Path) -> list[TimedChord]:
    """Load ground truth chords from JSON file (Harte notation)."""
    with path.open() as f:
        data = json.load(f)

    timed_chords: list[TimedChord] = []
    for item in data:
        label = item["chord"]
        start = float(item["start"])
        end = item.get("end")
        if end is not None:
            end = float(end)

        if label == "N":
            chord = None
        else:
            try:
                chord = from_harte(label)
            except ValueError:
                chord = None

        timed_chords.append(
            TimedChord(
                chord=chord,
                label=label,
                start=start,
                end=end,
            )
        )

    return timed_chords


def load_predicted_chords(path: Path) -> list[TimedChord]:
    """Load predicted chords from JSON file (pychord notation)."""
    with path.open() as f:
        data = json.load(f)

    timed_chords: list[TimedChord] = []
    for item in data:
        label = item["chord"]
        start = float(item["start"])
        end = item.get("end")
        if end is not None:
            end = float(end)

        if label == "N":
            chord = None
        else:
            try:
                chord = from_pychord(label)
            except ValueError:
                chord = None

        timed_chords.append(
            TimedChord(
                chord=chord,
                label=label,
                start=start,
                end=end,
            )
        )

    return timed_chords


def load_tab_data(tab_path: Path) -> tuple[tuple[TabChord, ...], list[str]]:
    """Load and parse tab sheet."""
    content = tab_path.read_text()

    # Strip frontmatter if present
    frontmatter_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = frontmatter_re.match(content)
    if match:
        content = content[match.end():]

    # Extract content from code blocks if present
    lines = content.split("\n")
    in_code_block = False
    tab_lines = []
    has_code_block = "```" in content

    if has_code_block:
        for line in lines:
            if line.strip() == "```":
                in_code_block = not in_code_block
                continue
            if in_code_block:
                tab_lines.append(line)
        tab_content = "\n".join(tab_lines)
    else:
        tab_content = content

    sheet = tab_parser.parse(tab_content)
    tab_chords = extract_tab_chords(sheet)
    sections = list(dict.fromkeys(tc.section for tc in tab_chords))

    return tab_chords, sections


def find_gt_chord_at_time(gt_chords: list[TimedChord], time: float) -> TimedChord | None:
    """Find the ground truth chord active at a given time."""
    for tc in gt_chords:
        if tc.start <= time and (tc.end is None or time < tc.end):
            return tc
    return None


def format_pitch_classes(chord: Chord | None) -> str:
    """Format pitch classes for display."""
    if chord is None:
        return "-"
    pcs = chord_to_pitch_classes(chord)
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return ", ".join(note_names[pc] for pc in sorted(pcs))


def to_pychord_label(chord: Chord | None, original_label: str) -> str:
    """Convert a chord to pychord notation for consistent display."""
    if chord is None:
        return original_label
    try:
        return chord.to_pychord()
    except (ValueError, Exception):
        return original_label


def main() -> None:
    """Run the alignment viewer app."""
    st.set_page_config(
        page_title="Chord Alignment Viewer",
        page_icon=":musical_note:",
        layout="wide",
    )

    st.title("Chord Alignment Viewer")

    # Paths
    project_root = Path(__file__).parent.parent

    # Allow selecting different datasets
    st.sidebar.header("Dataset")

    # Find available datasets (directories with required files)
    datasets = {}
    for d in project_root.iterdir():
        if d.is_dir():
            tab_file = d / "tab.md"
            pred_file = d / "pred_chords.json"
            if tab_file.exists() and pred_file.exists():
                datasets[d.name] = d

    if not datasets:
        st.error("No datasets found. Expected directories with tab.md and pred_chords.json files.")
        return

    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=list(datasets.keys()),
        index=0,
    )
    dataset_path = datasets[selected_dataset]

    # File paths
    tab_path = dataset_path / "tab.md"
    pred_path = dataset_path / "pred_chords.json"
    gt_path = dataset_path / "gt_chords.json"

    # Sidebar controls
    st.sidebar.header("Alignment Settings")

    min_similarity = st.sidebar.slider(
        "Minimum Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum similarity to accept a chord match",
    )

    lookahead = st.sidebar.slider(
        "Lookahead Window",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="Number of audio chords to search ahead",
    )

    use_transposition = st.sidebar.checkbox(
        "Auto-detect Transposition",
        value=True,
        help="Try all 12 transpositions and pick the best",
    )

    st.sidebar.header("Display Settings")

    color_by = st.sidebar.radio(
        "Color By",
        options=["Match Quality", "Section", "Pitch Class Similarity"],
        index=0,
    )

    show_unaligned = st.sidebar.checkbox(
        "Show Unaligned Tab Chords",
        value=True,
    )

    # Load data
    @st.cache_data
    def cached_load_tab(path: str) -> tuple[tuple[TabChord, ...], list[str]]:
        return load_tab_data(Path(path))

    @st.cache_data
    def cached_load_pred(path: str) -> list[TimedChord]:
        return load_predicted_chords(Path(path))

    @st.cache_data
    def cached_load_gt(path: str) -> list[TimedChord]:
        return load_ground_truth(Path(path))

    tab_chords, sections = cached_load_tab(str(tab_path))
    pred_chords = cached_load_pred(str(pred_path))

    gt_chords = None
    if gt_path.exists():
        gt_chords = cached_load_gt(str(gt_path))

    # Filter by section
    selected_sections = st.sidebar.multiselect(
        "Filter Sections",
        options=sections,
        default=sections,
        help="Show only chords from selected sections",
    )

    filtered_tab = tuple(tc for tc in tab_chords if tc.section in selected_sections)

    # Run alignment
    if use_transposition:
        result, best_transposition = align_with_transposition(
            tab_chords=list(filtered_tab),
            timed_chords=pred_chords,
            lookahead=lookahead,
            min_similarity=min_similarity,
        )
        transposition_info = f"Best transposition: {best_transposition} semitones"
    else:
        result = align_chords(
            tab_chords=list(filtered_tab),
            timed_chords=pred_chords,
            lookahead=lookahead,
            min_similarity=min_similarity,
        )
        transposition_info = "Transposition detection disabled"

    metrics = compute_alignment_metrics(result)

    # Display stats
    st.subheader("Alignment Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Tab Chords", len(filtered_tab))
    col2.metric("Predicted Chords", len(pred_chords))
    col3.metric("Aligned", f"{len(result.alignments)} ({metrics['coverage']:.0%})")
    col4.metric("Exact Matches", f"{metrics['exact_matches']:.0%}")
    col5.metric("Avg Similarity", f"{metrics['avg_similarity']:.2f}")

    st.caption(transposition_info)

    # Pitch class analysis
    st.subheader("Pitch Class Similarity Analysis")

    pc_col1, pc_col2, pc_col3 = st.columns(3)

    high_pc = sum(1 for a in result.alignments if chord_pitch_similarity(a.tab_chord.chord, a.timed_chord.chord) >= 0.8)
    med_pc = sum(1 for a in result.alignments if 0.5 <= chord_pitch_similarity(a.tab_chord.chord, a.timed_chord.chord) < 0.8)
    low_pc = sum(1 for a in result.alignments if chord_pitch_similarity(a.tab_chord.chord, a.timed_chord.chord) < 0.5)

    total = len(result.alignments) or 1
    pc_col1.metric("High PC Sim (>=0.8)", f"{high_pc} ({high_pc/total:.0%})")
    pc_col2.metric("Medium PC Sim", f"{med_pc} ({med_pc/total:.0%})")
    pc_col3.metric("Low PC Sim (<0.5)", f"{low_pc} ({low_pc/total:.0%})")

    # Ground truth comparison
    if gt_chords:
        st.subheader("Ground Truth Comparison")

        gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)
        gt_col1.metric("GT Chords", len(gt_chords))

        # Compare aligned chords to GT
        gt_exact = 0
        gt_root = 0
        gt_mismatch = 0

        for aligned in result.alignments:
            aligned_time = aligned.timed_chord.start
            gt_at_time = find_gt_chord_at_time(gt_chords, aligned_time)

            if gt_at_time is None or gt_at_time.chord is None:
                gt_mismatch += 1
                continue

            if aligned.tab_chord.chord is None:
                gt_mismatch += 1
                continue

            if roots_match(aligned.tab_chord.chord, gt_at_time.chord):
                if quality_category(aligned.tab_chord.chord.quality) == quality_category(gt_at_time.chord.quality):
                    gt_exact += 1
                else:
                    gt_root += 1
            else:
                gt_mismatch += 1

        gt_total = gt_exact + gt_root + gt_mismatch or 1
        gt_col2.metric("Exact vs GT", f"{gt_exact} ({gt_exact/gt_total:.0%})")
        gt_col3.metric("Root Match vs GT", f"{gt_root} ({gt_root/gt_total:.0%})")
        gt_col4.metric("Mismatch vs GT", f"{gt_mismatch} ({gt_mismatch/gt_total:.0%})")

    # Chord timeline visualization
    st.subheader("Chord Timeline")

    # Build timeline data
    timeline_data = []
    aligned_indices = {a.tab_chord.index for a in result.alignments}

    for aligned in result.alignments:
        tc = aligned.tab_chord
        timed = aligned.timed_chord

        # Determine color based on settings
        sim = chord_similarity(tc.chord, timed.chord)
        pc_sim = chord_pitch_similarity(tc.chord, timed.chord)

        if color_by == "Match Quality":
            if sim >= 1.0:
                color = MATCH_COLORS["exact"]
            elif sim >= 0.5:
                color = MATCH_COLORS["root_only"]
            else:
                color = MATCH_COLORS["mismatch"]
        elif color_by == "Pitch Class Similarity":
            # Gradient from red to green based on PC similarity
            if pc_sim >= 0.8:
                color = MATCH_COLORS["exact"]
            elif pc_sim >= 0.5:
                color = MATCH_COLORS["root_only"]
            else:
                color = MATCH_COLORS["mismatch"]
        else:  # Section
            color = get_section_color(tc.section, sections)

        # GT comparison
        gt_label = "-"
        gt_match = "-"
        if gt_chords:
            gt_at_time = find_gt_chord_at_time(gt_chords, timed.start)
            if gt_at_time:
                # Convert GT label to pychord format for consistent display
                gt_label = to_pychord_label(gt_at_time.chord, gt_at_time.label)
                if gt_at_time.chord and tc.chord:
                    if roots_match(tc.chord, gt_at_time.chord):
                        gt_match = "Root" if quality_category(tc.chord.quality) != quality_category(gt_at_time.chord.quality) else "Exact"
                    else:
                        gt_match = "Mismatch"

        timeline_data.append({
            "Section": tc.section,
            "Tab Chord": tc.label,
            "Matched": timed.label,
            "Start": f"{timed.start:.2f}s",
            "End": f"{timed.end:.2f}s" if timed.end else "-",
            "Similarity": f"{sim:.0%}",
            "PC Jaccard": f"{pc_sim:.2f}",
            "Tab PCs": format_pitch_classes(tc.chord),
            "Pred PCs": format_pitch_classes(timed.chord),
            "GT Chord": gt_label,
            "GT Match": gt_match,
            "_color": color,
        })

    # Add unaligned chords if requested
    if show_unaligned:
        for tc in filtered_tab:
            if tc.index not in aligned_indices:
                timeline_data.append({
                    "Section": tc.section,
                    "Tab Chord": tc.label,
                    "Matched": "(unaligned)",
                    "Start": "-",
                    "End": "-",
                    "Similarity": "-",
                    "PC Jaccard": "-",
                    "Tab PCs": format_pitch_classes(tc.chord),
                    "Pred PCs": "-",
                    "GT Chord": "-",
                    "GT Match": "-",
                    "_color": MATCH_COLORS["unaligned"],
                })

    # Display timeline
    if timeline_data:
        # Create a styled dataframe
        display_data = [{k: v for k, v in d.items() if not k.startswith("_")} for d in timeline_data]

        st.dataframe(
            display_data,
            use_container_width=True,
            height=400,
        )

    # Low similarity pairs detail
    with st.expander("Low Pitch Class Similarity Pairs", expanded=False):
        low_sim_data = []
        for aligned in result.alignments:
            pc_sim = chord_pitch_similarity(aligned.tab_chord.chord, aligned.timed_chord.chord)
            if pc_sim < 0.5:
                low_sim_data.append({
                    "Tab": aligned.tab_chord.label,
                    "Tab PCs": format_pitch_classes(aligned.tab_chord.chord),
                    "Predicted": aligned.timed_chord.label,
                    "Pred PCs": format_pitch_classes(aligned.timed_chord.chord),
                    "PC Jaccard": f"{pc_sim:.2f}",
                    "Section": aligned.tab_chord.section,
                })

        if low_sim_data:
            st.dataframe(low_sim_data, use_container_width=True)
        else:
            st.write("No low similarity pairs found.")

    # Section breakdown
    with st.expander("Section Breakdown", expanded=False):
        section_stats = {}
        for tc in filtered_tab:
            if tc.section not in section_stats:
                section_stats[tc.section] = {"total": 0, "aligned": 0}
            section_stats[tc.section]["total"] += 1

        for aligned in result.alignments:
            section = aligned.tab_chord.section
            if section in section_stats:
                section_stats[section]["aligned"] += 1

        section_data = []
        for section, stats in section_stats.items():
            pct = stats["aligned"] / stats["total"] * 100 if stats["total"] > 0 else 0
            section_data.append({
                "Section": section,
                "Total Chords": stats["total"],
                "Aligned": stats["aligned"],
                "Coverage": f"{pct:.0f}%",
            })

        st.dataframe(section_data, use_container_width=True)

    # Raw chord sequences
    with st.expander("Raw Chord Sequences", expanded=False):
        st.write("**Tab Chords:**")
        st.code(" ".join(tc.label for tc in filtered_tab[:50]) + ("..." if len(filtered_tab) > 50 else ""))

        st.write("**Predicted Chords:**")
        pred_labels = [tc.label for tc in pred_chords if tc.label != "N"]
        st.code(" ".join(pred_labels[:50]) + ("..." if len(pred_labels) > 50 else ""))

        if gt_chords:
            st.write("**Ground Truth Chords:**")
            gt_labels = [tc.label for tc in gt_chords if tc.label != "N"]
            st.code(" ".join(gt_labels[:50]) + ("..." if len(gt_labels) > 50 else ""))


if __name__ == "__main__":
    main()
