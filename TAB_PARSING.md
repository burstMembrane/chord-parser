# Tab Parsing Edge Cases

The tab format is not standardized. We need a robust parser that can handle and skip invalid content while extracting chord progressions.

## Invalid File Types (should be excluded)

### 1. Video-only files
Just contain a YouTube video ID, no actual chord data.
```
# zz-top-la-grange.md
np25pdAh4ok
```

### 2. Pure tablature (instrumental/lead)
No chord names, just fret positions. Common for instrumentals.
```
# santo-johnny-sleep-walk.md (instrumental)
e|-------------------5--5----5----5----|
B|-5--5----5---5-----5--5----5----5----|
G|-5--5----5---5------------------------|
```

### 3. Bass tabs
Bass notation (4 strings: G D A E), no chords.
```
# james-brown-cold-sweat.md
G|-----7--------7-----------------|
D|----------7---------------------|
A|5---5-----------5-------------3-|
E|------------------2---3-3-4-4---|
```

### 4. Wrong artist attribution
Files with generic artist names that don't match manifest.
- `artist: "Misc Cartoons"` - should be actual composer
- `artist: "Misc Soundtrack"` - should be actual artist
- `artist: "Misc Unsigned Bands"` - actual artist in title


## Valid Files with Parsing Challenges

### 1. Chord diagrams at top
Need to skip these, extract only chord names.
```
Chords:
G:     (32003X)
Cadd9: (X3203X)
D/F#:  (20023X)
```

### 2. Inline tablature mixed with chords
Has both chord progressions AND tab snippets. Extract chords, skip tabs.
```
# cream-sunshine-of-your-love.md
e|---------------------|
B|-3-3-1-3-------------|
G|---------2-1-0-------|

[Verse]
     D       C    D    A G F D  F \ D
It's getting near dawn,
```

### 3. Slash notation for rhythm
`F \ D` means slide from F to D - treat as two separate chords.

### 4. Chord-over-lyrics format
Standard format, parse chord line separately from lyric line.
```
D                          C                          D
Feelin' good, can't be real, must be dreamin' 'bout my drivin' wheel
```

### 5. Bar notation
Chord changes marked with `|` pipes.
```
| A/D   G/D | G/D | A/D   G/D | G/D |
```

### 6. Section markers
Skip `[Intro]`, `[Verse 1]`, `[Chorus]`, etc.

### 7. Repeat markers
`(x4)`, `(x2)`, `2x` indicate repeats.

### 8. Parenthetical chords
`(A/D)` - chord is optional or implied.

### 9. Coda/ending tabs
Some files have a small tab snippet at the very end for the coda.
```
[Coda]
e|2----0---3--------|
B|2--------0--------|
```

## Detection Heuristics

To classify a file:
1. **Video-only**: Body contains only alphanumeric string (YouTube ID)
2. **Pure tab**: Has `e|`, `B|`, `G|` lines but NO chord names like `Am`, `C`, `G7`
3. **Bass tab**: Has `G|`, `D|`, `A|`, `E|` (4-string) with no chord names
4. **Valid chord chart**: Contains chord names (uppercase letter + optional modifiers) over lyrics or in progressions
