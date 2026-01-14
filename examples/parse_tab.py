#!/usr/bin/env python3
"""CLI tool to parse tab files and export to JSON.

Usage:
    python examples/parse_tab.py <input_file> [output_file]

Examples:
    python examples/parse_tab.py testdata/upside_down.txt
    python examples/parse_tab.py testdata/upside_down.txt output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chord_parser import tab_parser
from chord_parser.tab_parser import Block, ChordLine, CommentLine, EmptyLine, LyricLine


def token_to_dict(token: tab_parser.Token) -> dict[str, Any]:
    """Convert a Token to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "text": token.text,
        "start": token.start,
        "end": token.end,
        "kind": token.kind,
    }
    if token.chord is not None:
        result["chord"] = {
            "root": token.chord.root,
            "quality": token.chord.quality,
            "bass": token.chord.bass,
            "harte": token.chord.to_harte(),
            "pychord": token.chord.to_pychord(),
        }
    return result


def block_to_dict(block: Block) -> dict[str, Any]:
    """Convert a Block to a JSON-serializable dict with chord-word sequences."""
    # Extract chord sequence
    chord_sequence = [
        {
            "text": token.text,
            "position": token.start,
            "chord": {
                "root": token.chord.root,
                "quality": token.chord.quality,
                "bass": token.chord.bass,
                "harte": token.chord.to_harte(),
                "pychord": token.chord.to_pychord(),
            },
        }
        for token in block.chord_tokens
        if token.kind == "chord" and token.chord is not None
    ]

    # Extract word sequence with attached chords
    word_sequence = []
    for word_attach in block.attachments.word_to_chords:
        word_data: dict[str, Any] = {
            "text": word_attach.word.text,
            "position": word_attach.word.start,
            "chords": [],
        }
        for chord_token in word_attach.chords:
            if chord_token.chord is not None:
                word_data["chords"].append({
                    "text": chord_token.text,
                    "harte": chord_token.chord.to_harte(),
                    "pychord": chord_token.chord.to_pychord(),
                })
        word_sequence.append(word_data)

    return {
        "type": "block",
        "chord_line": block.chord_raw.rstrip(),
        "lyric_line": block.lyric_raw.rstrip(),
        "width": block.width,
        "chord_sequence": chord_sequence,
        "word_sequence": word_sequence,
    }


def item_to_dict(item: tab_parser.Item) -> dict[str, Any]:
    """Convert any Item type to a JSON-serializable dict."""
    if isinstance(item, Block):
        return block_to_dict(item)

    if isinstance(item, ChordLine):
        chords = [
            {
                "text": token.text,
                "position": token.start,
                "chord": {
                    "root": token.chord.root,
                    "quality": token.chord.quality,
                    "bass": token.chord.bass,
                    "harte": token.chord.to_harte(),
                    "pychord": token.chord.to_pychord(),
                },
            }
            for token in item.tokens
            if token.kind == "chord" and token.chord is not None
        ]
        return {
            "type": "chord_line",
            "raw": item.raw.rstrip(),
            "chords": chords,
        }

    if isinstance(item, LyricLine):
        words = [
            {"text": t.text, "position": t.start}
            for t in item.tokens
            if t.kind == "word"
        ]
        return {
            "type": "lyric_line",
            "raw": item.raw.rstrip(),
            "words": words,
        }

    if isinstance(item, CommentLine):
        return {
            "type": "comment",
            "raw": item.raw.rstrip(),
        }

    if isinstance(item, EmptyLine):
        return {"type": "empty"}

    return {"type": "unknown"}


def section_to_dict(section: tab_parser.Section) -> dict[str, Any]:
    """Convert a Section to a JSON-serializable dict."""
    return {
        "name": section.name,
        "items": [item_to_dict(item) for item in section.items],
    }


def tab_sheet_to_dict(sheet: tab_parser.TabSheet) -> dict[str, Any]:
    """Convert a TabSheet to a JSON-serializable dict."""
    return {
        "sections": [section_to_dict(s) for s in sheet.sections],
    }


def parse_tab_file(input_path: Path) -> dict[str, Any]:
    """Parse a tab file and return JSON-serializable data."""
    text = input_path.read_text()
    sheet = tab_parser.parse(text)
    return tab_sheet_to_dict(sheet)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse a tab file and export to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s testdata/upside_down.txt
  %(prog)s testdata/upside_down.txt -o output.json
  %(prog)s testdata/simple_pair.txt --pretty
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input tab file to parse",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        data = parse_tab_file(args.input)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    json_output = json.dumps(data, indent=indent, ensure_ascii=False)

    if args.output:
        args.output.write_text(json_output)
        print(f"Wrote output to {args.output}")
    else:
        print(json_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
