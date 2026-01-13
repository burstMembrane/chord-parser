# Harte Library Documentation

A Python extension of music21 for working with chords encoded in Harte Notation.

**Version:** 0.4.5
**License:** MIT
**GitHub:** https://github.com/andreamust/harte-library
**PyPI:** https://pypi.org/project/harte-library/

## Installation

```bash
pip install harte-library --upgrade
```

Requires Python >= 3.7

## Dependencies

- `music21` - Base music library
- `Lark` - Parsing library

## Core Features

1. **Interoperability** - Integrate Harte-notated chords with music21
2. **Interpretability** - Interpret Harte chords including shorthand unrolling
3. **Simplification** - Standardize chords using prettify functionality

## API Reference

### HarteInterval Class

Converts Harte interval notation to music21 intervals.

```python
from harte.interval import HarteInterval

interval = HarteInterval('b6')
interval.name           # Get interval name
interval.isConsonant()  # Check if consonant
```

### Harte Class

Main class for chord processing. Extends music21 chord with Harte-specific methods.

```python
from harte.harte import Harte

chord = Harte('C#:maj7(b6)/b3')
chord.bass()      # E
chord.root()      # C#
chord.fullName    # Full chord name
```

### Methods

| Method | Description |
|--------|-------------|
| `get_degrees()` | Retrieve chord intervals (excluding shorthand) |
| `get_midi_pitches()` | Get ordered MIDI pitches |
| `get_root()` | Get root as string |
| `get_bass()` | Calculate root-to-bass interval |
| `contains_shorthand()` | Check for shorthand notation |
| `get_shorthand()` | Get shorthand representation |
| `unwrap_shorthand()` | Expand all intervals including shorthand |
| `prettify()` | Decompose and recompose using concise shorthand |

## Usage Examples

### Basic Chord Creation

```python
from harte.harte import Harte

# Simple major chord
chord = Harte('C:maj')

# Complex chord with alterations and bass
chord = Harte('C#:maj7(b6)/b3')
print(chord.bass())   # E
print(chord.root())   # C#
```

### Working with Intervals

```python
from harte.interval import HarteInterval

interval = HarteInterval('b6')
print(interval.name)
print(interval.isConsonant())
```

### Prettification

Convert verbose chord notation to concise shorthand:

```python
chord = Harte('D:(b3,5,7,9)')
pretty = chord.prettify()  # D:minmaj7(9)
```

### Unwrapping Shorthand

Expand shorthand notation to full interval list:

```python
chord = Harte('C:maj7')
intervals = chord.unwrap_shorthand()
```

## Harte Notation Reference

Harte notation uses the format: `ROOT:QUALITY(ALTERATIONS)/BASS`

- **ROOT**: Note name (C, D, E, F, G, A, B) with optional accidentals (#, b)
- **QUALITY**: Chord quality shorthand (maj, min, dim, aug, etc.)
- **ALTERATIONS**: Additional intervals in parentheses (b6, #9, etc.)
- **BASS**: Bass note as interval from root, prefixed with /

### Common Shorthand Qualities

- `maj` - Major triad (1, 3, 5)
- `min` - Minor triad (1, b3, 5)
- `dim` - Diminished triad (1, b3, b5)
- `aug` - Augmented triad (1, 3, #5)
- `maj7` - Major seventh (1, 3, 5, 7)
- `min7` - Minor seventh (1, b3, 5, b7)
- `7` - Dominant seventh (1, 3, 5, b7)
- `dim7` - Diminished seventh (1, b3, b5, bb7)
- `hdim7` - Half-diminished seventh (1, b3, b5, b7)
- `minmaj7` - Minor-major seventh (1, b3, 5, 7)

## Related Projects

- **ChoCo Dataset**: Collection of 20,000+ timed chord annotations
- **music21**: MIT music toolkit for Python
