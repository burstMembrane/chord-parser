"""High-level API for chord alignment.

This module provides simplified API functions for common alignment workflows.
"""

from __future__ import annotations

import re
from pathlib import Path

from chord_parser.alignment.aligner import align_chords
from chord_parser.alignment.extractor import extract_tab_chords
from chord_parser.alignment.models import AlignmentResult
from chord_parser.alignment.smams import parse_chordino_from_smams
from chord_parser.tab_parser.parser import parse


# Pattern to extract YAML frontmatter from markdown
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    """Strip YAML frontmatter from markdown text.

    Parameters
    ----------
    text : str
        Markdown text potentially with frontmatter.

    Returns
    -------
    str
        Text with frontmatter removed.
    """
    match = FRONTMATTER_RE.match(text)
    if match:
        return text[match.end() :]
    return text


def align_tab_with_smams(
    smams_path: str | Path,
    tab_path: str | Path,
    *,
    lookahead: int = 5,
    min_similarity: float = 0.5,
) -> AlignmentResult:
    """Align tab sheet chords with chordino annotations from a SMAMS file.

    This is the primary high-level API for chord alignment. It handles:
    - Loading and parsing the SMAMS file to extract chordino annotations
    - Loading and parsing the tab markdown file (with optional frontmatter)
    - Extracting chord sequences from both sources
    - Running greedy forward alignment

    Parameters
    ----------
    smams_path : str | Path
        Path to the SMAMS file containing chordino annotations.
    tab_path : str | Path
        Path to the tab markdown file. The file may contain YAML frontmatter
        (between --- delimiters) which will be stripped before parsing.
    lookahead : int, optional
        Number of timed chords to search ahead for each tab chord.
        Default is 5.
    min_similarity : float, optional
        Minimum similarity score (0.0 to 1.0) to accept a match.
        Default is 0.5.

    Returns
    -------
    AlignmentResult
        The alignment result containing:
        - alignments: Matched chord pairs with distances
        - tab_chords: All chords from the tab
        - timed_chords: All chords from chordino
        - total_distance: Sum of distances for all matches
        - normalized_distance: Average distance per match

    Raises
    ------
    FileNotFoundError
        If either file does not exist.
    ValueError
        If the SMAMS file has no chordino annotations.

    Examples
    --------
    >>> from chord_parser.alignment import align_tab_with_smams
    >>> result = align_tab_with_smams("song.smams", "song_tab.md")
    >>> print(f"Aligned {len(result.alignments)} chords")
    >>> for aligned in result.alignments[:5]:
    ...     print(f"{aligned.tab_chord.label} -> {aligned.timed_chord.label} "
    ...           f"(dist={aligned.distance:.2f}, t={aligned.get_start():.2f}s)")
    """
    # Load chordino annotations from SMAMS
    timed_chords = parse_chordino_from_smams(smams_path)

    # Load and parse tab file
    tab_path = Path(tab_path)
    tab_text = tab_path.read_text()

    # Strip frontmatter if present
    tab_content = _strip_frontmatter(tab_text)

    # Parse tab sheet
    tab_sheet = parse(tab_content)

    # Extract tab chords
    tab_chords = extract_tab_chords(tab_sheet)

    # Run alignment
    return align_chords(
        tab_chords=tab_chords,
        timed_chords=timed_chords,
        lookahead=lookahead,
        min_similarity=min_similarity,
    )


def align_tab_text_with_smams(
    smams_path: str | Path,
    tab_text: str,
    *,
    lookahead: int = 5,
    min_similarity: float = 0.5,
) -> AlignmentResult:
    """Align tab sheet text with chordino annotations from a SMAMS file.

    Same as align_tab_with_smams but accepts tab content as a string
    instead of a file path.

    Parameters
    ----------
    smams_path : str | Path
        Path to the SMAMS file containing chordino annotations.
    tab_text : str
        Tab sheet content as a string. May contain YAML frontmatter.
    lookahead : int, optional
        Number of timed chords to search ahead for each tab chord.
        Default is 5.
    min_similarity : float, optional
        Minimum similarity score (0.0 to 1.0) to accept a match.
        Default is 0.5.

    Returns
    -------
    AlignmentResult
        The alignment result.

    See Also
    --------
    align_tab_with_smams : File-based version of this function.
    """
    # Load chordino annotations from SMAMS
    timed_chords = parse_chordino_from_smams(smams_path)

    # Strip frontmatter if present
    tab_content = _strip_frontmatter(tab_text)

    # Parse tab sheet
    tab_sheet = parse(tab_content)

    # Extract tab chords
    tab_chords = extract_tab_chords(tab_sheet)

    # Run alignment
    return align_chords(
        tab_chords=tab_chords,
        timed_chords=timed_chords,
        lookahead=lookahead,
        min_similarity=min_similarity,
    )
