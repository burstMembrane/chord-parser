#!/usr/bin/env python3
"""Generate a large catalogue of chord-symbol strings (lead-sheet style) and write to JSON.

This is not "all chords that can exist" (infinite grammar), but it is an exhaustive set
for a configurable grammar: roots x qualities x extensions x alterations x adds x sus x slash bass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


# ----------------------------
# Configurable chord grammar
# ----------------------------

ROOTS = [
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
]

# Qualities that pychord supports as standalone
QUALITIES = [
    "",  # major triad (e.g., C)
    "m",  # minor (Cm)
    "dim",  # diminished triad (Cdim)
    "aug",  # augmented triad (Caug)
    "sus2",  # Csus2
    "sus4",  # Csus4
    "5",  # power chord (C5)
]

# Extensions that work with empty quality (major base)
# These are standalone qualities in pychord, not combinable with QUALITIES
EXTENSIONS = [
    "",  # just the quality itself
    "6",  # C6 (major 6th)
    "7",  # C7 (dominant 7th)
    "maj7",  # Cmaj7
    "m7",  # Cm7 (minor 7th) - standalone, not "m" + "7"
    "dim7",  # Cdim7 - standalone, not "dim" + "7"
    "m7-5",  # Cm7-5 (half-diminished)
    "9",  # C9
    "maj9",  # Cmaj9
    "m9",  # Cm9
    "11",  # C11
    "m11",  # Cm11
    "13",  # C13
    "add9",  # Cadd9
    "m6",  # Cm6
    "mmaj7",  # Cmmaj7 (minor-major 7th)
]

# Alterations (mostly for dominant/extended chords)
ALTERATIONS = [
    "",
    "b5",
    "#5",
    "b9",
    "#9",
    "b11",
    "#11",
    "b13",
    "#13",
]

# Allow up to N alterations per chord (2 is already huge)
MAX_ALTERATIONS = 2

# Adds (triads and some 7ths)
ADDS = [
    "",
    "add2",
    "add4",
    "add9",
    "add11",
    "add13",
]

# Optional "no" tones (rare but seen)
NOS = [
    "",
    "no3",
    "no5",
]

# Special symbol for no-chord
NO_CHORD_SYMBOLS = ["N"]


@dataclass(frozen=True)
class ChordEntry:
    """A single chord entry in the catalogue."""

    symbol: str


def _powerset_alterations(alts: list[str], max_k: int) -> list[tuple[str, ...]]:
    """All alteration combinations up to max_k, excluding empty handled separately."""
    combos: list[tuple[str, ...]] = [()]
    for k in range(1, max_k + 1):
        for tpl in product(alts, repeat=k):
            # prevent duplicates like ("b9","b9")
            if len(set(tpl)) != len(tpl):
                continue
            combos.append(tuple(sorted(tpl)))
    # unique
    return list(dict.fromkeys(combos))


def generate_chord_symbols(  # noqa: PLR0912, PLR0913
    roots: list[str] | None = None,
    qualities: list[str] | None = None,
    extensions: list[str] | None = None,
    adds: list[str] | None = None,
    nos: list[str] | None = None,
    alterations: list[str] | None = None,
    max_alterations: int = MAX_ALTERATIONS,
    include_slash_bass: bool = True,
    include_no_chord: bool = True,
) -> list[str]:
    """Generate chord symbol strings.

    Notes
    -----
    This is lead-sheet-ish text, not Harte.
    We do some basic sanity filtering to avoid nonsense like "Csus2m9".
    """
    roots = roots if roots is not None else ROOTS
    qualities = qualities if qualities is not None else QUALITIES
    extensions = extensions if extensions is not None else EXTENSIONS
    adds = adds if adds is not None else ADDS
    nos = nos if nos is not None else NOS
    alterations = alterations if alterations is not None else ALTERATIONS

    alt_combos = _powerset_alterations([a for a in alterations if a], max_alterations)

    out: set[str] = set()

    for root, qual, ext, add, no, alt_tpl in product(roots, qualities, extensions, adds, nos, alt_combos):
        # Basic filters: extensions are standalone pychord qualities, not combinable
        # Only empty quality ("") can combine with extensions
        # Non-empty qualities (m, dim, aug, sus2, sus4, 5) are standalone
        if qual != "" and ext != "":
            continue

        # Power chords "5" are standalone only
        if qual == "5" and add not in ("", ):
            continue

        # Build symbol:
        # - If ext is minor-like (m7 etc.), keep qual empty unless you're using "m" + "7" style.
        base = f"{root}{qual}"
        if ext:
            # If ext already includes an "m" (m7/m9/...), don't also add "m" quality
            if ext.startswith("m") and qual == "m":
                continue
            base = f"{root}{qual}{ext}"

        # Apply add/no/alterations
        parts = [base]

        # Alterations get appended directly (C7b9#11)
        for a in alt_tpl:
            if a:
                parts.append(a)

        if add:
            parts.append(add)

        if no:
            parts.append(no)

        symbol = "".join(parts)

        # Additional cleanup: avoid "Cmaj7add9" being "Cmaj7add9" (fine) but
        # avoid "Cadd9add11" etc. (we allow only one add from the list)
        out.add(symbol)

        # Slash bass variants
        if include_slash_bass and symbol not in NO_CHORD_SYMBOLS:
            for bass in roots:
                # avoid trivial "C/C"
                if bass == root:
                    continue
                out.add(f"{symbol}/{bass}")

    if include_no_chord:
        out.update(NO_CHORD_SYMBOLS)

    # Stable ordering
    return sorted(out)


def write_json(path: Path, symbols: Iterable[str]) -> None:
    """Write chord symbols to JSON file."""
    payload: dict[str, object] = {
        "schema": "chord-symbol-catalogue/v1",
        "count": 0,
        "chords": [],
    }
    chords = [asdict(ChordEntry(symbol=s)) for s in symbols]
    payload["count"] = len(chords)
    payload["chords"] = chords

    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main() -> None:
    """Generate chord catalogue and write to JSON."""
    out_dir = Path(__file__).parent.parent / "testdata"

    # Generate core catalogue (no slash bass, no alterations) for testing
    core_symbols = generate_chord_symbols(
        include_slash_bass=False,
        alterations=[""],
        max_alterations=0,
        nos=[""],
        adds=[""],
    )
    core_path = out_dir / "chord_catalogue_core.json"
    write_json(core_path, core_symbols)
    print(f"Wrote {len(core_symbols)} core chord symbols to {core_path.resolve()}")

    # Generate full catalogue (with slash bass) - warning: very large!
    # Uncomment if needed:
    # full_symbols = generate_chord_symbols()
    # full_path = out_dir / "chord_catalogue_full.json"
    # write_json(full_path, full_symbols)
    # print(f"Wrote {len(full_symbols)} full chord symbols to {full_path.resolve()}")


if __name__ == "__main__":
    main()
