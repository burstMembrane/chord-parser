"""Chord parser library for converting between chord notation formats.

This library provides tools for parsing and converting chords between
pychord's simplified notation (e.g., "Gm7") and Harte notation (e.g., "G:min7").

Examples
--------
>>> from chord_parser import Chord, from_pychord, from_harte

>>> # Parse from pychord notation
>>> chord = from_pychord("Gm7")
>>> chord.to_harte()
'G:min7'

>>> # Parse from Harte notation
>>> chord = from_harte("G:min7")
>>> chord.to_pychord()
'Gm7'

>>> # Direct conversion functions
>>> from chord_parser import pychord_to_harte, harte_to_pychord
>>> pychord_to_harte("Bbm7")
'Bb:min7'
>>> harte_to_pychord("C:maj")
'C'
"""

from chord_parser.converter import (
    from_harte,
    from_pychord,
    harte_quality_to_pychord,
    harte_to_pychord,
    pychord_quality_to_harte,
    pychord_to_harte,
)
from chord_parser.models import Chord

__all__ = [
    "Chord",
    "from_harte",
    "from_pychord",
    "harte_quality_to_pychord",
    "harte_to_pychord",
    "pychord_quality_to_harte",
    "pychord_to_harte",
]
