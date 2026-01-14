"""Column-aware tokenizer for tab sheets.

This module provides tokenization that preserves column span information,
which is essential for chord-to-word alignment in monospace text.
"""

from chord_parser.tab_parser.models import Token


def tokenize_line(line: str) -> list[Token]:
    """Tokenize a line preserving column spans.

    Splits on whitespace while tracking the start and end column positions
    of each token. Does not strip the line, preserving whitespace semantics.

    Parameters
    ----------
    line : str
        The line to tokenize. Should not include newline characters.

    Returns
    -------
    list[Token]
        List of tokens with text, start (inclusive), end (exclusive),
        and kind set to "other" (classification happens later).

    Examples
    --------
    >>> tokens = tokenize_line("Gm     C")
    >>> [(t.text, t.start, t.end) for t in tokens]
    [('Gm', 0, 2), ('C', 7, 8)]

    >>> tokens = tokenize_line("Hello  world")
    >>> [(t.text, t.start, t.end) for t in tokens]
    [('Hello', 0, 5), ('world', 7, 12)]
    """
    tokens: list[Token] = []
    i = 0
    n = len(line)

    while i < n:
        # Skip whitespace
        if line[i].isspace():
            i += 1
            continue

        # Start of a token
        start = i

        # Capture maximal non-whitespace substring
        while i < n and not line[i].isspace():
            i += 1

        end = i
        text = line[start:end]

        tokens.append(
            Token(
                text=text,
                start=start,
                end=end,
                kind="other",  # Classification happens in chord_detector
            )
        )

    return tokens
