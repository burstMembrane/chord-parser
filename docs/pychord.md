# PyChord Documentation

A Python library for handling musical chords.

**Version:** 1.2.2
**License:** MIT
**GitHub:** https://github.com/yuma-m/pychord
**Documentation:** https://pychord.readthedocs.io/en/latest/
**PyPI:** https://pypi.org/project/pychord/

## Installation

```bash
pip install pychord
```

Requires Python >= 3.6

## Core Classes

### Chord

Main class for handling individual chords.

```python
from pychord import Chord

c = Chord("Am7")
c.info()  # Display chord details
```

#### Constructor

```python
Chord(chord: str)
```

Creates a chord from string notation (e.g., "C", "Am7", "F#m7-5/A").

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `chord` | str | Full chord string |
| `root` | str | Root note |
| `quality` | Quality | Chord quality object |
| `appended` | list | Appended notes |
| `on` | str | Bass note (for slash chords) |

#### Methods

**`components(visible=True)`**

Returns component notes as strings or integer positions.

```python
c = Chord("Am7")
c.components()  # ['A', 'C', 'E', 'G']
```

**`components_with_pitch(root_pitch)`**

Returns pitched note names.

```python
c = Chord("Am7")
c.components_with_pitch(root_pitch=3)  # ['A3', 'C4', 'E4', 'G4']
```

**`transpose(trans, scale='C')`**

Transposes the chord by semitone count.

```python
c = Chord("Am7/G")
c.transpose(3)  # Cm7/Bb
```

**`from_note_index(note, quality, scale, diatonic=False, chromatic=0)`** (class method)

Creates chords from scale degrees.

```python
Chord.from_note_index(1, "maj7", "C")  # Creates Cmaj7
```

### ChordProgression

Manages sequences of chords.

```python
from pychord import ChordProgression

cp = ChordProgression(["C", "G/B", "Am"])
cp.transpose(+3)  # Transpose entire progression
```

#### Methods

| Method | Description |
|--------|-------------|
| `append(chord)` | Add a chord to the progression |
| `insert(index, chord)` | Insert chord at specified position |
| `pop(index=-1)` | Remove and return a chord |
| `transpose(trans)` | Transpose all chords |

#### Properties

| Property | Description |
|----------|-------------|
| `chords` | List of component chords |

### Quality

Represents chord qualities (e.g., major, minor seventh).

**`get_components(root='C', visible=False)`**

Retrieves constituent intervals.

**`append_on_chord(on_chord, root)`**

Creates slash chord variations.

### QualityManager

Singleton managing available chord qualities.

```python
from pychord import QualityManager

qm = QualityManager()
qm.get_quality("maj7")
qm.set_quality("custom", (0, 4, 7, 10))
```

#### Methods

| Method | Description |
|--------|-------------|
| `get_quality(name, inversion=0)` | Retrieve quality definitions |
| `find_quality_from_components(components)` | Identify quality from intervals |
| `set_quality(name, components)` | Register new chord qualities |

## Utility Functions

### Analyzer Module

```python
from pychord import find_chords_from_notes

find_chords_from_notes(["C", "E", "G"])  # Identifies matching chords
```

| Function | Description |
|----------|-------------|
| `find_chords_from_notes(notes)` | Identify possible chords from note list |
| `notes_to_positions(notes, root)` | Convert notes to interval positions |

### Parser Module

```python
from pychord.parser import parse

parse("Am7/G")  # Extract root, quality, appended notes, bass
```

### Utils Module

| Function | Description |
|----------|-------------|
| `note_to_val(note)` | Convert note name to numeric index (C=0, B=11) |
| `val_to_note(val, scale)` | Convert numeric index to note name |
| `transpose_note(note, transpose, scale)` | Transpose individual notes |

## Usage Examples

### Creating and Analyzing Chords

```python
from pychord import Chord

# Create a chord
c = Chord("Am7")
c.info()

# Get components
print(c.components())       # ['A', 'C', 'E', 'G']
print(c.root)               # A
print(c.quality)            # m7

# With pitch information
print(c.components_with_pitch(root_pitch=3))  # ['A3', 'C4', 'E4', 'G4']
```

### Transposition

```python
from pychord import Chord

c = Chord("Am7/G")
c.transpose(3)
print(c)  # Cm7/Bb

# Transpose down
c.transpose(-5)
```

### Chord Comparison

PyChord recognizes enharmonic equivalents:

```python
Chord("C#") == Chord("Db")  # True (if comparing pitches)
```

### Finding Chords from Notes

```python
from pychord import find_chords_from_notes

chords = find_chords_from_notes(["C", "E", "G"])
print(chords)  # [<Chord: C>]

# More complex
chords = find_chords_from_notes(["A", "C", "E", "G"])
print(chords)  # [<Chord: Am7>]
```

### Chord Progressions

```python
from pychord import ChordProgression

# Create a progression
cp = ChordProgression(["C", "G/B", "Am", "F"])

# Transpose the entire progression
cp.transpose(+3)  # Now: Eb, Bb/D, Cm, Ab

# Manipulate the progression
cp.append(Chord("G"))
cp.insert(0, Chord("C"))
removed = cp.pop()
```

### Working with Inversions

```python
from pychord import Chord

# Slash notation for inversions
c = Chord("C/E")    # C major, first inversion
c = Chord("C/G")    # C major, second inversion

# Numeric inversion notation
c = Chord("C/1")    # First inversion
c = Chord("C/2")    # Second inversion
```

### Custom Chord Qualities

```python
from pychord import QualityManager

qm = QualityManager()

# Define a custom quality
qm.set_quality("add2", (0, 2, 4, 7))  # Root, 2nd, 3rd, 5th

# Now you can use it
c = Chord("Cadd2")
```

## Module Structure

```
pychord/
    __init__.py
    analyzer.py      # Chord analysis functions
    chord.py         # Chord class
    parser.py        # Chord notation parser
    progression.py   # ChordProgression class
    quality.py       # Quality and QualityManager
    utils.py         # Utility functions
    constants/
        qualities.py # Built-in chord qualities
        scales.py    # Scale definitions
```
