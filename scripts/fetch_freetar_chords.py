#!/usr/bin/env python3
"""Fetch chord data from FreeTar for the Billboard 100 dataset.

This script reads the Billboard 100 manifest and runs freetar-cli
to fetch chord data for each song.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load the Billboard 100 manifest.

    Parameters
    ----------
    manifest_path
        Path to the manifest.json file.

    Returns
    -------
    list[dict]
        List of song entries from the manifest.
    """
    with open(manifest_path) as f:
        return json.load(f)


def fetch_chords(title: str, artist: str, output_path: Path, use_curl: bool = True) -> bool:
    """Fetch chords for a song using freetar-cli.

    Parameters
    ----------
    title
        Song title.
    artist
        Artist name.
    output_path
        Path to save the output JSON.
    use_curl
        Whether to use curl for HTTP requests.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    query = f"{title} {artist}"
    cmd = ["freetar-cli", "get-best", query]
    if use_curl:
        cmd.append("--use-curl")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.stdout)
            return True
        else:
            print(f"  Error: {result.stderr.strip()}", file=sys.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"  Timeout fetching chords", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: freetar-cli not found. Is it installed?", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Run the chord fetching script."""
    parser = argparse.ArgumentParser(
        description="Fetch chord data from FreeTar for Billboard 100 dataset"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("billboard_100/manifest.json"),
        help="Path to manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("billboard_100/freetar"),
        help="Directory to save freetar chord files",
    )
    parser.add_argument(
        "--no-curl",
        action="store_true",
        help="Don't use curl for HTTP requests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip songs that already have chord files",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"Loaded {len(manifest)} songs from manifest")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, song in enumerate(manifest, 1):
        song_id = song["id"]
        title = song["title"]
        artist = song["artist"]
        output_path = args.output_dir / f"{song_id}.json"

        if args.skip_existing and output_path.exists():
            print(f"[{i}/{len(manifest)}] Skipping {title} - {artist} (exists)")
            skip_count += 1
            continue

        print(f"[{i}/{len(manifest)}] Fetching: {title} - {artist}")

        if args.dry_run:
            query = f"{title} {artist}"
            curl_flag = "" if args.no_curl else " --use-curl"
            print(f'  freetar-cli get-best "{query}"{curl_flag} > {output_path}')
            continue

        if fetch_chords(title, artist, output_path, use_curl=not args.no_curl):
            print(f"  Saved to {output_path}")
            success_count += 1
        else:
            fail_count += 1

    print(f"\nComplete: {success_count} success, {skip_count} skipped, {fail_count} failed")


if __name__ == "__main__":
    main()
