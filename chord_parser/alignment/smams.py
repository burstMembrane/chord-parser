"""SMAMS file parser for word-aligned lyrics.

This module extracts word-aligned lyrics from SMAMS (SheetMuse Audio Metadata Schema)
files, providing timing information for vocal word onsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlignedWord:
    """A word with timing information from vocal alignment.

    Parameters
    ----------
    word : str
        The word text.
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    """

    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Get the word duration in seconds."""
        return self.end - self.start


@dataclass(frozen=True)
class WordAlignedLyrics:
    """Collection of word-aligned lyrics from a SMAMS file.

    Parameters
    ----------
    words : tuple[AlignedWord, ...]
        All aligned words in temporal order.
    source_path : str | None
        Path to the source SMAMS file.
    """

    words: tuple[AlignedWord, ...]
    source_path: str | None = None

    def find_word(self, query: str, start_from: float = 0.0) -> AlignedWord | None:
        """Find the first occurrence of a word after a given time.

        Parameters
        ----------
        query : str
            The word to search for (case-insensitive).
        start_from : float
            Only return words starting at or after this time.

        Returns
        -------
        AlignedWord | None
            The matched word or None if not found.
        """
        query_lower = _normalize_word(query)
        for w in self.words:
            if w.start >= start_from and _normalize_word(w.word) == query_lower:
                return w
        return None

    def find_all_words(self, query: str) -> list[AlignedWord]:
        """Find all occurrences of a word.

        Parameters
        ----------
        query : str
            The word to search for (case-insensitive).

        Returns
        -------
        list[AlignedWord]
            All matching words in temporal order.
        """
        query_lower = _normalize_word(query)
        return [w for w in self.words if _normalize_word(w.word) == query_lower]


def _normalize_word(word: str) -> str:
    """Normalize a word for matching.

    Strips punctuation, apostrophes, and converts to lowercase.

    Parameters
    ----------
    word : str
        The word to normalize.

    Returns
    -------
    str
        Normalized word.
    """
    # Remove leading/trailing punctuation and apostrophes
    normalized = word.lower().strip()
    # Remove common punctuation
    for char in ".,!?;:'\"()-":
        normalized = normalized.replace(char, "")
    return normalized


def _collect_word_aligned_lyrics(obj: Any) -> list[dict[str, Any]]:
    """Recursively collect word_aligned_lyrics observations from SMAMS data.

    Parameters
    ----------
    obj : Any
        The SMAMS data object to search.

    Returns
    -------
    list[dict[str, Any]]
        All word alignment observations found.
    """
    results: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        if obj.get("namespace") == "word_aligned_lyrics":
            obs = obj.get("data", {}).get("observations", [])
            results.extend(obs)
        for v in obj.values():
            results.extend(_collect_word_aligned_lyrics(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_word_aligned_lyrics(item))

    return results


def parse_smams(path: str | Path) -> WordAlignedLyrics:
    """Parse a SMAMS file and extract word-aligned lyrics.

    Parameters
    ----------
    path : str | Path
        Path to the SMAMS file.

    Returns
    -------
    WordAlignedLyrics
        The extracted word alignments.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(path)
    with path.open() as f:
        data = json.load(f)

    observations = _collect_word_aligned_lyrics(data)

    words: list[AlignedWord] = []
    for obs in observations:
        interval = obs.get("interval", {})
        time = interval.get("time", 0.0)
        duration = interval.get("duration", 0.0)
        value = obs.get("value", "")

        if value:  # Skip empty words
            words.append(
                AlignedWord(
                    word=value,
                    start=time,
                    end=time + duration,
                )
            )

    # Sort by start time
    words.sort(key=lambda w: w.start)

    return WordAlignedLyrics(
        words=tuple(words),
        source_path=str(path),
    )


def parse_smams_from_dict(data: dict[str, Any]) -> WordAlignedLyrics:
    """Parse word-aligned lyrics from an already-loaded SMAMS dict.

    Parameters
    ----------
    data : dict[str, Any]
        The parsed SMAMS data.

    Returns
    -------
    WordAlignedLyrics
        The extracted word alignments.
    """
    observations = _collect_word_aligned_lyrics(data)

    words: list[AlignedWord] = []
    for obs in observations:
        interval = obs.get("interval", {})
        time = interval.get("time", 0.0)
        duration = interval.get("duration", 0.0)
        value = obs.get("value", "")

        if value:
            words.append(
                AlignedWord(
                    word=value,
                    start=time,
                    end=time + duration,
                )
            )

    words.sort(key=lambda w: w.start)

    return WordAlignedLyrics(words=tuple(words))
