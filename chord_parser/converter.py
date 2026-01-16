"""Chord notation converter between pychord and Harte formats.

This module provides conversion functions between pychord's simplified
chord notation (e.g., "Gm7") and Harte notation (e.g., "G:min7").
"""

from chord_parser.models import Chord


def _normalize_bass(bass: str | None) -> str | None:
    """Normalize bass note, converting empty strings to None."""
    return bass if bass else None


# Mapping from pychord quality names to Harte shorthand
PYCHORD_TO_HARTE_QUALITY: dict[str, str] = {
    "": "maj",
    "m": "min",
    "m7": "min7",
    "7": "7",
    "maj7": "maj7",
    "M7": "maj7",
    "dim": "dim",
    "dim7": "dim7",
    "dim6": "dim6",
    "aug": "aug",
    "aug7": "aug7",
    "m7-5": "hdim7",
    "m7b5": "hdim7",
    "sus4": "sus4",
    "sus2": "sus2",
    "7sus4": "7sus4",
    "7sus2": "7sus2",
    "sus47": "sus4(b7)",
    "sus27": "sus2(b7)",
    "add9": "maj(9)",
    "madd9": "min(9)",
    "9": "9",
    "m9": "min9",
    "maj9": "maj9",
    "11": "11",
    "m11": "min11",
    "maj11": "maj11",
    "13": "13",
    "m13": "min13",
    "maj13": "maj13",
    "6": "maj6",
    "m6": "min6",
    "mmaj7": "minmaj7",
    "mM7": "minmaj7",
    "5": "5",
}

# Reverse mapping from Harte shorthand to pychord quality
HARTE_TO_PYCHORD_QUALITY: dict[str, str] = {
    "maj": "",
    "min": "m",
    "min7": "m7",
    "7": "7",
    "maj7": "maj7",
    "dim": "dim",
    "dim7": "dim7",
    "dim6": "dim6",
    "aug": "aug",
    "aug7": "aug7",
    "hdim7": "m7-5",
    "sus4": "sus4",
    "sus2": "sus2",
    "7sus4": "7sus4",
    "7sus2": "7sus2",
    "sus4(b7)": "sus47",
    "sus2(b7)": "sus27",
    "maj(9)": "add9",
    "min(9)": "madd9",
    "9": "9",
    "min9": "m9",
    "maj9": "maj9",
    "11": "11",
    "min11": "m11",
    "maj11": "maj11",
    "13": "13",
    "min13": "m13",
    "maj13": "maj13",
    "maj6": "6",
    "min6": "m6",
    "minmaj7": "mmaj7",
    "5": "5",
}


def pychord_quality_to_harte(pychord_quality: str) -> str:
    """Convert a pychord quality string to Harte shorthand.

    Parameters
    ----------
    pychord_quality : str
        The pychord quality (e.g., "m7", "maj7", "dim").

    Returns
    -------
    str
        The equivalent Harte shorthand (e.g., "min7", "maj7", "dim").

    Raises
    ------
    ValueError
        If the quality is not recognized.

    Examples
    --------
    >>> pychord_quality_to_harte("m7")
    'min7'
    >>> pychord_quality_to_harte("")
    'maj'
    """
    if pychord_quality in PYCHORD_TO_HARTE_QUALITY:
        return PYCHORD_TO_HARTE_QUALITY[pychord_quality]
    msg = f"Unknown pychord quality: {pychord_quality}"
    raise ValueError(msg)


def harte_quality_to_pychord(harte_quality: str) -> str:
    """Convert a Harte shorthand to pychord quality string.

    Parameters
    ----------
    harte_quality : str
        The Harte shorthand (e.g., "min7", "maj7", "dim").

    Returns
    -------
    str
        The equivalent pychord quality (e.g., "m7", "maj7", "dim").

    Raises
    ------
    ValueError
        If the quality is not recognized.

    Examples
    --------
    >>> harte_quality_to_pychord("min7")
    'm7'
    >>> harte_quality_to_pychord("maj")
    ''
    """
    if harte_quality in HARTE_TO_PYCHORD_QUALITY:
        return HARTE_TO_PYCHORD_QUALITY[harte_quality]
    msg = f"Unknown Harte quality: {harte_quality}"
    raise ValueError(msg)


def from_pychord(chord_str: str) -> Chord:
    """Parse a pychord notation string into a Chord object.

    Parameters
    ----------
    chord_str : str
        Chord in pychord notation (e.g., "Gm7", "C", "F#dim7/A").

    Returns
    -------
    Chord
        Unified chord representation.

    Examples
    --------
    >>> chord = from_pychord("Gm7")
    >>> chord.root
    'G'
    >>> chord.quality
    'min7'
    >>> chord.to_harte()
    'G:min7'
    """
    from pychord import Chord as PyChord

    pc = PyChord(chord_str)
    quality_name = str(pc.quality)
    harte_quality = pychord_quality_to_harte(quality_name)

    return Chord(
        root=pc.root,
        quality=harte_quality,
        bass=_normalize_bass(pc.on),
    )


def from_harte(chord_str: str) -> Chord:
    """Parse a Harte notation string into a Chord object.

    Parameters
    ----------
    chord_str : str
        Chord in Harte notation (e.g., "G:min7", "C:maj", "F#:dim7/A").

    Returns
    -------
    Chord
        Unified chord representation.

    Examples
    --------
    >>> chord = from_harte("G:min7")
    >>> chord.root
    'G'
    >>> chord.quality
    'min7'
    >>> chord.to_pychord()
    'Gm7'
    """
    from harte.harte import Harte

    hc = Harte(chord_str)
    root = hc.get_root()
    shorthand = hc.get_shorthand()

    # Handle bass note if present
    bass = None
    if "/" in chord_str:
        bass_part = chord_str.split("/")[-1]
        bass = bass_part

    return Chord(
        root=root,
        quality=shorthand if shorthand else "maj",
        bass=bass,
    )


def pychord_to_harte(chord_str: str) -> str:
    """Convert a pychord notation string to Harte notation.

    Parameters
    ----------
    chord_str : str
        Chord in pychord notation (e.g., "Gm7", "C", "Bbm").

    Returns
    -------
    str
        Chord in Harte notation (e.g., "G:min7", "C:maj", "Bb:min").

    Examples
    --------
    >>> pychord_to_harte("Gm7")
    'G:min7'
    >>> pychord_to_harte("C")
    'C:maj'
    """
    chord = from_pychord(chord_str)
    return chord.to_harte()


def harte_to_pychord(chord_str: str) -> str:
    """Convert a Harte notation string to pychord notation.

    Parameters
    ----------
    chord_str : str
        Chord in Harte notation (e.g., "G:min7", "C:maj").

    Returns
    -------
    str
        Chord in pychord notation (e.g., "Gm7", "C").

    Examples
    --------
    >>> harte_to_pychord("G:min7")
    'Gm7'
    >>> harte_to_pychord("C:maj")
    'C'
    """
    chord = from_harte(chord_str)
    return chord.to_pychord()
