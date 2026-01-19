"""DECIBEL HMM-based chord alignment module.

This module provides HMM-based alignment of tab chords to audio using
the DECIBEL library's jump alignment algorithm. It supports automatic
transposition detection and handles section/line boundaries for improved
alignment quality.

Example
-------
>>> from chord_parser.alignment import extract_tab_chords
>>> from chord_parser.decibel import align_tab_with_audio
>>> from chord_parser.tab_parser import parse_tab
>>>
>>> tab = parse_tab(tab_text)
>>> tab_chords = list(extract_tab_chords(tab))
>>> result = align_tab_with_audio(tab_chords, "song.mp3")
>>> print(f"Transposition: {result.transposition} semitones")
>>> for aligned in result.alignment_result.alignments:
...     print(f"{aligned.tab_chord.label}: {aligned.get_start():.2f}s")
"""

# Re-export DECIBEL types that users may need
from decibel.audio_tab_aligner.hmm_parameters import HMMParameters
from decibel.music_objects.chord_vocabulary import ChordVocabulary

from chord_parser.decibel.adapter import (
    chord_to_decibel,
    chord_to_harte_string,
    tabchord_to_chord_id,
    tabchords_to_alignment_arrays,
    timedchord_to_chord_id,
)
from chord_parser.decibel.alignment import (
    HMMAlignmentResult,
    align_tab_with_audio,
    align_tab_with_chroma,
    create_default_hmm_parameters,
)
from chord_parser.decibel.features import AudioFeatures, extract_audio_features
from chord_parser.decibel.training import (
    load_billboard_chords,
    load_billboard_dataset,
    load_gt_chords_from_json,
    load_hmm_parameters,
    save_hmm_parameters,
    train_hmm,
)

__all__ = [
    "AudioFeatures",
    "ChordVocabulary",
    "HMMAlignmentResult",
    "HMMParameters",
    "align_tab_with_audio",
    "align_tab_with_chroma",
    "chord_to_decibel",
    "chord_to_harte_string",
    "create_default_hmm_parameters",
    "extract_audio_features",
    "load_billboard_chords",
    "load_billboard_dataset",
    "load_gt_chords_from_json",
    "load_hmm_parameters",
    "save_hmm_parameters",
    "tabchord_to_chord_id",
    "tabchords_to_alignment_arrays",
    "timedchord_to_chord_id",
    "train_hmm",
]
