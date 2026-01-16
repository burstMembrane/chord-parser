#!/usr/bin/env python
"""Streamlit app for viewing SMAMS chord alignment with tab sheets.

Run with: uv run streamlit run examples/smams_alignment_viewer.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from smams import SMAMS
from streamlit_wavesurfer import Region, WaveSurferOptions, wavesurfer

from chord_parser import from_pychord
from chord_parser.alignment import (
    TimedChord,
    align_chords,
    chord_similarity,
    extract_tab_chords,
)
from chord_parser.alignment.models import TabChord
from chord_parser.models import Chord
from chord_parser.tab_parser import parse, parse_chords

SIMILARITY_PRESETS = {
    "Standard (0.5)": 0.5,
    "Strict (0.8)": 0.8,
    "Relaxed (0.3)": 0.3,
}

MATCH_COLORS = {
    "perfect": "rgba(34, 197, 94, 0.4)",
    "close": "rgba(250, 204, 21, 0.4)",
    "mismatch": "rgba(239, 68, 68, 0.4)",
}

SMAMS_COLORS = {
    "default": "rgba(59, 130, 246, 0.4)",
    "N": "rgba(156, 163, 175, 0.3)",
}


def load_timed_chords(smams_path: Path) -> list[TimedChord]:
    """Load timed chords from SMAMS file."""
    smams_file = SMAMS.load(str(smams_path))
    chords_ns = smams_file.get_namespace("chords")

    return [
        TimedChord(
            chord=from_pychord(str(obs.value)) if obs.value != "N" else None,
            start=obs.interval.time,
            end=obs.interval.time + obs.interval.duration,
            label=obs.value or "N",
        )
        for obs in chords_ns.get_observations()
    ]


def get_match_color(similarity: float, threshold: float) -> str:
    """Get color based on chord similarity."""
    if similarity == 1.0:
        return MATCH_COLORS["perfect"]
    elif similarity >= threshold:
        return MATCH_COLORS["close"]
    return MATCH_COLORS["mismatch"]


def main() -> None:
    """Run the alignment viewer."""
    st.set_page_config(page_title="Chord Alignment Viewer", layout="wide")
    st.title("Chord Alignment Viewer")

    # Default paths
    project_root = Path(__file__).parent.parent
    default_smams = project_root / "wonderwall" / "wonderwall.smams"
    default_tab = project_root / "wonderwall" / "debug_tab.txt"

    # Sidebar - file selection
    st.sidebar.header("Files")
    smams_path = Path(st.sidebar.text_input("SMAMS Path", str(default_smams)))
    tab_path = Path(st.sidebar.text_input("Tab Path", str(default_tab)))

    # Validate files
    if not smams_path.exists():
        st.error(f"SMAMS file not found: {smams_path}")
        return
    if not tab_path.exists():
        st.error(f"Tab file not found: {tab_path}")
        return

    # Find audio files
    audio_files = {f.stem: f for ext in [".mp3", ".wav", ".ogg"] for f in smams_path.parent.glob(f"*{ext}")}
    if not audio_files:
        st.error(f"No audio files found in {smams_path.parent}")
        return

    audio_choice = st.sidebar.selectbox("Audio File", list(audio_files.keys()))
    audio_path = audio_files[audio_choice]

    # Alignment settings
    st.sidebar.header("Settings")
    similarity_preset = st.sidebar.selectbox("Similarity Threshold", list(SIMILARITY_PRESETS.keys()))
    min_similarity = SIMILARITY_PRESETS[similarity_preset]
    lookahead = st.sidebar.slider("Lookahead Window", 3, 20, 10)
    threshold = st.sidebar.slider("Close Match Display Threshold", 0.0, 1.0, 0.5, 0.05)

    # Load data
    timed_chords = load_timed_chords(smams_path)

    # Parse tab and create TabChord objects for alignment
    raw_chords = parse_chords(tab_path.read_text())
    tab_chords = [
        TabChord(chord=c, label=c.to_pychord(), section="Tab", index=i)
        for i, c in enumerate(raw_chords)
    ]

    st.sidebar.markdown(f"**Tab chords:** {len(tab_chords)}")
    st.sidebar.markdown(f"**SMAMS chords:** {len(timed_chords)}")

    # Run alignment
    result = align_chords(tab_chords, timed_chords, lookahead=lookahead, min_similarity=min_similarity)

    # Build regions and stats
    regions: list[Region] = []
    stats = {"perfect": 0, "close": 0, "mismatch": 0}

    for aligned in result.alignments:
        similarity = 1.0 - aligned.distance
        color = get_match_color(similarity, threshold)

        if similarity == 1.0:
            stats["perfect"] += 1
        elif similarity >= threshold:
            stats["close"] += 1
        else:
            stats["mismatch"] += 1

        regions.append(
            Region(
                start=aligned.timed_chord.start,
                end=aligned.timed_chord.end or aligned.timed_chord.start + 0.5,
                content=f"{aligned.tab_chord.label}",
                color=color,
            )
        )

    # Sort regions by start time for cleaner display
    regions.sort(key=lambda r: r.start)

    # Stats display
    st.subheader("Alignment Statistics")
    col1, col2, col3, col4 = st.columns(4)
    total = len(tab_chords)
    matched = len(result.alignments)
    unmatched = total - matched
    col1.metric("Tab Chords", total)
    col2.metric("Matched", f"{matched} ({100 * matched // total}%)" if total else "0")
    col3.metric("Perfect", stats["perfect"])
    col4.metric("Unmatched", unmatched)

    # Aligned waveform
    st.subheader("Aligned Tab Chords")
    wavesurfer(
        audio_src=str(audio_path),
        regions=regions,
        plugins=["regions", "timeline", "zoom"],
        show_controls=True,
        region_colormap="viridis",
        wave_options=WaveSurferOptions(minPxPerSec=100),
        key="aligned",
    )

    st.markdown(
        "**Legend:** "
        '<span style="background-color: rgba(34, 197, 94, 0.6); padding: 2px 8px;">Perfect</span> '
        '<span style="background-color: rgba(250, 204, 21, 0.6); padding: 2px 8px;">Close</span> '
        '<span style="background-color: rgba(239, 68, 68, 0.6); padding: 2px 8px;">Mismatch</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # SMAMS chords waveform
    st.subheader("SMAMS Detected Chords")
    smams_regions = [
        Region(
            start=tc.start,
            end=tc.end or tc.start + 0.5,
            content=tc.label,
            color=SMAMS_COLORS["N"] if tc.label == "N" else SMAMS_COLORS["default"],
        )
        for tc in timed_chords
    ]

    wavesurfer(
        audio_src=str(audio_path),
        regions=smams_regions,
        plugins=["regions", "timeline", "zoom"],
        show_controls=True,
        region_colormap="plasma",
        wave_options=WaveSurferOptions(minPxPerSec=100),
        key="smams",
    )

    # Details table
    with st.expander("Alignment Details"):
        st.dataframe(
            [
                {
                    "Tab": a.tab_chord.label,
                    "Matched": a.timed_chord.label,
                    "Start": f"{a.timed_chord.start:.2f}s",
                    "Similarity": f"{1.0 - a.distance:.0%}",
                }
                for a in sorted(result.alignments, key=lambda x: x.timed_chord.start)
            ],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
