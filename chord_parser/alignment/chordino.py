"""Chordino annotation loader.

This module provides functionality to load and parse chord annotations
from Chordino JSON output files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chord_parser.models import Chord

from chord_parser.alignment.models import TimedChord
from chord_parser.converter import from_pychord


def parse_chordino_chord(label: str) -> Chord | None:
    """Parse a Chordino chord label to a Chord object.

    Parameters
    ----------
    label : str
        Chord label from Chordino (e.g., "Gm7", "C", "N").

    Returns
    -------
    Chord | None
        Parsed Chord, or None if label is "N" (no chord).

    Raises
    ------
    ValueError
        If the chord cannot be parsed.
    """
    if label == "N":
        return None

    return from_pychord(label)


def load_chordino_json(path: str | Path) -> list[TimedChord]:
    """Load Chordino annotations from a JSON file.

    Expected JSON structure:
    {
        "chords": [
            {"start": 0.0, "end": 1.0, "chord": "Gm"},
            ...
        ]
    }

    Parameters
    ----------
    path : str | Path
        Path to the Chordino JSON file.

    Returns
    -------
    list[TimedChord]
        List of timed chord annotations.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    return parse_chordino_data(data)


def parse_chordino_data(data: dict[str, Any]) -> list[TimedChord]:
    """Parse Chordino data from a dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Chordino data with a "chords" key.

    Returns
    -------
    list[TimedChord]
        List of timed chord annotations.
    """
    chords_data = data.get("chords", [])
    timed_chords: list[TimedChord] = []

    for item in chords_data:
        label = item["chord"]
        start = float(item["start"])
        end = item.get("end")
        if end is not None:
            end = float(end)

        try:
            chord = parse_chordino_chord(label)
        except ValueError:
            # Skip unparseable chords but log them
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
