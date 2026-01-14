#!/usr/bin/env python
"""Streamlit app for viewing chord alignment with audio waveform.

Run with: uv run streamlit run examples/alignment_viewer.py

Features:
- Live Chordino analysis with configurable parameters
- DTW alignment between tab chords and Chordino output
- Comparison against ground truth (Billboard annotations)
- Visual waveform with aligned chord regions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from streamlit_wavesurfer import Region, WaveSurferOptions, wavesurfer

from chord_parser import tab_parser
from chord_parser.alignment import (
    AlignedChord,
    align_chords,
    chord_distance_exact,
    chord_distance_flexible,
    chord_distance_pitchclass,
    chord_to_pitch_class_set,
    extract_tab_chords,
    load_chordino_json,
)
from chord_parser.alignment.models import TimedChord
from chord_parser.converter import from_harte, from_pychord
from chord_parser.models import Chord


class TuningMode(Enum):
    """Chordino tuning mode."""

    LOCAL = 0
    GLOBAL = 1


@dataclass
class ChordinoParams:
    """Chordino plugin parameters."""

    use_nnls: bool = True
    roll_on: float = 0.0
    tuning_mode: TuningMode = TuningMode.GLOBAL
    spectral_whitening: float = 1.0
    spectral_shape: float = 0.7
    boost_n_likelihood: float = 0.1

    def to_dict(self) -> dict[str, float]:
        """Convert to parameter dictionary for vamprust."""
        return {
            "useNNLS": 1.0 if self.use_nnls else 0.0,
            "rollon": self.roll_on,
            "tuningmode": float(self.tuning_mode.value),
            "whitening": self.spectral_whitening,
            "s": self.spectral_shape,
            "boostn": self.boost_n_likelihood,
        }


# Genre-specific configurations
CHORDINO_CONFIGS: dict[str, ChordinoParams] = {
    "Default": ChordinoParams(),
    "Pop/Rock": ChordinoParams(
        use_nnls=True,
        roll_on=0.0,
        tuning_mode=TuningMode.GLOBAL,
        spectral_whitening=0.8,
        spectral_shape=0.6,
        boost_n_likelihood=0.02,
    ),
    "Jazz": ChordinoParams(
        use_nnls=True,
        roll_on=0.5,
        tuning_mode=TuningMode.LOCAL,
        spectral_whitening=0.9,
        spectral_shape=0.8,
        boost_n_likelihood=0.05,
    ),
    "Classical": ChordinoParams(
        use_nnls=True,
        roll_on=0.0,
        tuning_mode=TuningMode.GLOBAL,
        spectral_whitening=0.7,
        spectral_shape=0.5,
        boost_n_likelihood=0.1,
    ),
    "Electronic": ChordinoParams(
        use_nnls=True,
        roll_on=1.0,
        tuning_mode=TuningMode.GLOBAL,
        spectral_whitening=0.95,
        spectral_shape=0.7,
        boost_n_likelihood=0.01,
    ),
}

# Distance function registry
DISTANCE_FUNCTIONS = {
    "Flexible (root + quality)": chord_distance_flexible,
    "Exact (full match)": chord_distance_exact,
    "Pitch Class (Jaccard)": chord_distance_pitchclass,
}

# Color scheme for match quality
MATCH_COLORS = {
    "perfect": "rgba(34, 197, 94, 0.4)",  # green
    "close": "rgba(250, 204, 21, 0.4)",  # yellow
    "mismatch": "rgba(239, 68, 68, 0.4)",  # red
}

# Section colors for visual distinction
SECTION_COLORS = [
    "rgba(99, 102, 241, 0.4)",  # indigo
    "rgba(236, 72, 153, 0.4)",  # pink
    "rgba(14, 165, 233, 0.4)",  # sky
    "rgba(168, 85, 247, 0.4)",  # purple
    "rgba(20, 184, 166, 0.4)",  # teal
    "rgba(249, 115, 22, 0.4)",  # orange
]


def get_match_color(distance: float, threshold: float) -> str:
    """Get color based on alignment distance."""
    if distance == 0.0:
        return MATCH_COLORS["perfect"]
    elif distance <= threshold:
        return MATCH_COLORS["close"]
    return MATCH_COLORS["mismatch"]


def get_section_color(section: str, sections: list[str]) -> str:
    """Get color for a section."""
    if section in sections:
        idx = sections.index(section)
        return SECTION_COLORS[idx % len(SECTION_COLORS)]
    return SECTION_COLORS[0]


def format_pitch_classes(chord: Chord | None) -> str:
    """Format pitch classes for display."""
    if chord is None:
        return "-"
    try:
        pc_set = chord_to_pitch_class_set(chord, include_bass=False)
        return str(sorted(pc_set))
    except (ValueError, Exception):
        return "-"


def run_chordino(audio_path: Path, params: ChordinoParams) -> list[TimedChord]:
    """Run Chordino with specified parameters using vamprust."""
    from vamprust._vamprust import PyVampHost as VampHost

    # Load audio
    samples, sr = sf.read(audio_path, dtype="float32")
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)

    host = VampHost()

    # Find and load Chordino plugin
    libraries = host.find_plugin_libraries()
    chordino_lib = None
    for lib_path in libraries:
        if "chordino" in lib_path.lower() or "nnls-chroma" in lib_path.lower():
            chordino_lib = host.load_library(lib_path)
            if chordino_lib:
                break

    if chordino_lib is None:
        raise RuntimeError("Could not find Chordino/NNLS-Chroma plugin")

    # Find Chordino plugin
    plugins = chordino_lib.list_plugins()
    chordino_idx = None
    for i, plugin_info in enumerate(plugins):
        if "chordino" in plugin_info.identifier.lower():
            chordino_idx = i
            break

    if chordino_idx is None:
        raise RuntimeError("Could not find chordino plugin in library")

    # Instantiate and configure plugin
    plugin = chordino_lib.instantiate_plugin(chordino_idx, float(sr))
    if plugin is None:
        raise RuntimeError("Failed to instantiate Chordino plugin")

    plugin.set_parameters(params.to_dict())
    plugin.initialize(float(sr), channels=1)

    # Process audio
    features = plugin.process_audio_full(
        samples.tolist(),
        float(sr),
        channels=1,
        output_index=0,
    )

    # Parse features into TimedChord objects
    chords: list[TimedChord] = []
    if features:
        for i, feature in enumerate(features):
            sec = feature.get("sec", 0)
            nsec = feature.get("nsec", 0)
            start = sec + nsec / 1_000_000_000
            label = feature.get("label", "N")

            if i + 1 < len(features):
                next_sec = features[i + 1].get("sec", 0)
                next_nsec = features[i + 1].get("nsec", 0)
                end = next_sec + next_nsec / 1_000_000_000
            else:
                end = start + 0.5

            # Parse chord
            chord_obj = None
            if label != "N":
                try:
                    chord_obj = from_pychord(label)
                except (ValueError, Exception):
                    pass

            chords.append(TimedChord(start=start, end=end, chord=chord_obj, label=label))

    return chords


def load_ground_truth(json_path: Path) -> list[TimedChord]:
    """Load ground truth from JSON file."""
    import json

    with open(json_path) as f:
        data = json.load(f)

    chords = []
    for entry in data["ground_truth"]["chords"]:
        chord_obj = None
        label = entry["chord"]
        if label != "N":
            try:
                chord_obj = from_harte(label)
            except (ValueError, Exception):
                pass

        chords.append(
            TimedChord(
                start=entry["start"],
                end=entry["end"],
                chord=chord_obj,
                label=label,
            )
        )
    return chords


def alignment_to_regions(
    alignments: tuple[AlignedChord, ...],
    color_by: str,
    threshold: float,
    sections: list[str],
) -> list[Region]:
    """Convert alignment results to wavesurfer regions."""
    regions = []
    seen_times: set[float] = set()

    for aligned in alignments:
        start = aligned.timed_chord.start

        if start in seen_times:
            continue
        seen_times.add(start)

        end = aligned.timed_chord.end or start + 0.5
        content = aligned.tab_chord.label
        section = aligned.tab_chord.section

        if color_by == "Match Quality":
            color = get_match_color(aligned.distance, threshold)
        else:
            color = get_section_color(section, sections)

        regions.append(
            Region(
                start=start,
                end=end,
                content=content,
                color=color,
            )
        )

    return regions


def compute_gt_accuracy(
    alignments: tuple[AlignedChord, ...],
    ground_truth: list[TimedChord],
    threshold: float = 0.25,
) -> dict[str, float]:
    """Compare aligned tab chords against ground truth."""
    perfect = 0
    close = 0
    mismatch = 0

    for aligned in alignments:
        tab_chord = aligned.tab_chord.chord
        timed = aligned.timed_chord

        if tab_chord is None:
            continue

        # Find GT chord at this time
        gt_chord = None
        for gt in ground_truth:
            if gt.chord is None:
                continue
            overlap_start = max(timed.start, gt.start)
            overlap_end = min(timed.end or timed.start + 0.5, gt.end)
            if overlap_end > overlap_start:
                gt_chord = gt.chord
                break

        if gt_chord is None:
            continue

        dist = chord_distance_pitchclass(tab_chord, gt_chord)
        if dist == 0.0:
            perfect += 1
        elif dist <= threshold:
            close += 1
        else:
            mismatch += 1

    total = perfect + close + mismatch
    return {
        "perfect": perfect,
        "close": close,
        "mismatch": mismatch,
        "total": total,
        "accuracy": (perfect + close) / total if total > 0 else 0,
    }


def load_tab_data(tab_path: Path) -> tuple:
    """Load and parse tab sheet."""
    content = tab_path.read_text()

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
    sections = list(dict.fromkeys(tc.section for tc in tab_chords))

    return tab_chords, sheet, sections


def main() -> None:
    """Run the alignment viewer app."""
    st.set_page_config(
        page_title="Chord Alignment Viewer",
        page_icon="musical_note",
        layout="wide",
    )

    st.title("Chord Alignment Viewer")

    # Paths
    project_root = Path(__file__).parent.parent
    tab_path = project_root / "upsidedown" / "tab.md"
    gt_path = project_root / "upsidedown" / "1285_chordino.json"

    # Audio options
    audio_files = {
        "Mix (with vocals)": project_root / "upsidedown" / "1285.ogg",
        "Instrumental": project_root / "upsidedown" / "1285_instrumental.wav",
    }

    # Check files exist
    if not tab_path.exists():
        st.error(f"Tab file not found: {tab_path}")
        return

    # Sidebar controls
    st.sidebar.header("Audio Source")

    audio_choice = st.sidebar.selectbox(
        "Audio File",
        options=list(audio_files.keys()),
        index=1,  # Default to instrumental
        help="Choose between mix or instrumental audio",
    )
    audio_path = audio_files[audio_choice]

    if not audio_path.exists():
        st.error(f"Audio file not found: {audio_path}")
        return

    st.sidebar.header("Chordino Settings")

    chordino_config = st.sidebar.selectbox(
        "Preset Configuration",
        options=list(CHORDINO_CONFIGS.keys()),
        index=1,  # Default to Pop/Rock
        help="Genre-specific Chordino parameters",
    )

    # Show current params
    params = CHORDINO_CONFIGS[chordino_config]
    with st.sidebar.expander("Parameter Details"):
        st.write(f"- NNLS: {params.use_nnls}")
        st.write(f"- Roll-on: {params.roll_on}")
        st.write(f"- Tuning: {params.tuning_mode.name}")
        st.write(f"- Whitening: {params.spectral_whitening}")
        st.write(f"- Shape: {params.spectral_shape}")
        st.write(f"- Boost N: {params.boost_n_likelihood}")

    st.sidebar.header("Alignment Settings")

    distance_fn_name = st.sidebar.selectbox(
        "Distance Function",
        options=list(DISTANCE_FUNCTIONS.keys()),
        index=0,
        help="How to measure chord similarity",
    )
    distance_fn = DISTANCE_FUNCTIONS[distance_fn_name]

    threshold = st.sidebar.slider(
        "Close Match Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Distance threshold for 'close' matches",
    )

    color_by = st.sidebar.radio(
        "Color Regions By",
        options=["Match Quality", "Section"],
        index=0,
    )

    # Load tab data
    @st.cache_data
    def cached_load_tab():
        return load_tab_data(tab_path)

    tab_chords, sheet, sections = cached_load_tab()

    # Filter by section
    selected_sections = st.sidebar.multiselect(
        "Filter Sections",
        options=sections,
        default=sections,
        help="Show only chords from selected sections",
    )

    # Run Chordino (cached by audio path and config)
    @st.cache_data
    def cached_chordino(audio_path_str: str, config_name: str):
        params = CHORDINO_CONFIGS[config_name]
        return run_chordino(Path(audio_path_str), params)

    with st.spinner("Running Chordino analysis..."):
        timed_chords = cached_chordino(str(audio_path), chordino_config)

    # Run alignment
    @st.cache_data
    def cached_align(audio_path_str: str, config_name: str, fn_name: str):
        timed = cached_chordino(audio_path_str, config_name)
        tab, _, _ = cached_load_tab()
        fn = DISTANCE_FUNCTIONS[fn_name]
        return align_chords(tab, timed, distance_fn=fn)

    result = cached_align(str(audio_path), chordino_config, distance_fn_name)

    # Filter alignments by section
    filtered_alignments = tuple(
        a for a in result.alignments if a.tab_chord.section in selected_sections
    )

    # Load ground truth for comparison
    gt_chords = None
    gt_accuracy = None
    if gt_path.exists():
        @st.cache_data
        def cached_load_gt():
            return load_ground_truth(gt_path)

        gt_chords = cached_load_gt()
        gt_accuracy = compute_gt_accuracy(filtered_alignments, gt_chords, threshold)

    # Stats
    st.subheader("Alignment Statistics")

    col1, col2, col3, col4 = st.columns(4)

    perfect = sum(1 for a in filtered_alignments if a.distance == 0.0)
    close = sum(1 for a in filtered_alignments if 0 < a.distance <= threshold)
    mismatch = sum(1 for a in filtered_alignments if a.distance > threshold)
    total = len(filtered_alignments)

    col1.metric("Tab Chords", total)
    col2.metric(
        "Perfect (Tab-Chordino)",
        f"{perfect} ({100*perfect/total:.0f}%)" if total else "0",
    )
    col3.metric(
        "Close (Tab-Chordino)",
        f"{close} ({100*close/total:.0f}%)" if total else "0",
    )
    col4.metric(
        "Mismatch (Tab-Chordino)",
        f"{mismatch} ({100*mismatch/total:.0f}%)" if total else "0",
    )

    # Show Chordino info
    st.caption(
        f"Chordino detected {len(timed_chords)} chords using '{chordino_config}' preset on {audio_choice}"
    )

    # Convert to regions
    regions = alignment_to_regions(filtered_alignments, color_by, threshold, sections)

    # Waveform display
    st.subheader("Audio Waveform with Aligned Chords")

    wavesurfer(
        audio_src=str(audio_path),
        regions=regions,
        plugins=["regions", "timeline", "zoom"],
        show_controls=True,
        region_colormap="viridis",
        wave_options=WaveSurferOptions(minPxPerSec=200),
        key="waveform",
    )

    # Legend
    st.markdown("---")
    if color_by == "Match Quality":
        st.markdown(
            "**Legend:** "
            '<span style="background-color: rgba(34, 197, 94, 0.6); padding: 2px 8px;">Perfect Match</span> '
            '<span style="background-color: rgba(250, 204, 21, 0.6); padding: 2px 8px;">Close Match</span> '
            '<span style="background-color: rgba(239, 68, 68, 0.6); padding: 2px 8px;">Mismatch</span>',
            unsafe_allow_html=True,
        )

    # Alignment table
    with st.expander("Alignment Details", expanded=False):
        table_data = []
        for a in filtered_alignments:
            table_data.append(
                {
                    "Tab Chord": a.tab_chord.label,
                    "Tab PC": format_pitch_classes(a.tab_chord.chord),
                    "Chordino": a.timed_chord.label,
                    "Chordino PC": format_pitch_classes(a.timed_chord.chord),
                    "Start": f"{a.timed_chord.start:.2f}s",
                    "End": f"{a.timed_chord.end:.2f}s" if a.timed_chord.end else "-",
                    "Distance": f"{a.distance:.2f}",
                    "Section": a.tab_chord.section,
                }
            )
        st.dataframe(table_data, use_container_width=True)

    # Ground truth comparison
    if gt_chords and gt_accuracy:
        with st.expander("Ground Truth Comparison", expanded=False):
            st.write(f"**Billboard Ground Truth:** {len(gt_chords)} annotations")
            st.write(f"**Tab vs GT Results:**")
            st.write(f"- Perfect matches: {gt_accuracy['perfect']}")
            st.write(f"- Close matches: {gt_accuracy['close']}")
            st.write(f"- Mismatches: {gt_accuracy['mismatch']}")
            st.write(f"- **Accuracy: {gt_accuracy['accuracy']*100:.1f}%**")


if __name__ == "__main__":
    main()
