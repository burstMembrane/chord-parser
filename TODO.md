read https://mir-eval.readthedocs.io/latest/api/chord.html


### Metrics

mir_eval.chord.root(): Only compares the root of the chords.

mir_eval.chord.majmin(): Only compares major, minor, and “no chord” labels.

mir_eval.chord.majmin_inv(): Compares major/minor chords, with inversions. The bass note must exist in the triad.

mir_eval.chord.mirex(): A estimated chord is considered correct if it shares at least three pitch classes in common.

mir_eval.chord.thirds(): Chords are compared at the level of major or minor thirds (root and third), For example, both (‘A:7’, ‘A:maj’) and (‘A:min’, ‘A:dim’) are equivalent, as the third is major and minor in quality, respectively.

mir_eval.chord.thirds_inv(): Same as above, with inversions (bass relationships).

mir_eval.chord.triads(): Chords are considered at the level of triads (major, minor, augmented, diminished, suspended), meaning that, in addition to the root, the quality is only considered through #5th scale degree (for augmented chords). For example, (‘A:7’, ‘A:maj’) are equivalent, while (‘A:min’, ‘A:dim’) and (‘A:aug’, ‘A:maj’) are not.

mir_eval.chord.triads_inv(): Same as above, with inversions (bass relationships).

mir_eval.chord.tetrads(): Chords are considered at the level of the entire quality in closed voicing, i.e. spanning only a single octave; extended chords (9’s, 11’s and 13’s) are rolled into a single octave with any upper voices included as extensions. For example, (‘A:7’, ‘A:9’) are equivalent but (‘A:7’, ‘A:maj7’) are not.

mir_eval.chord.tetrads_inv(): Same as above, with inversions (bass relationships).

mir_eval.chord.sevenths(): Compares according to MIREX “sevenths” rules; that is, only major, major seventh, seventh, minor, minor seventh and no chord labels are compared.

mir_eval.chord.sevenths_inv(): Same as above, with inversions (bass relationships).

mir_eval.chord.overseg(): Computes the level of over-segmentation between estimated and reference intervals.

mir_eval.chord.underseg(): Computes the level of under-segmentation between estimated and reference intervals.

mir_eval.chord.seg(): Computes the minimum of over- and under-segmentation between estimated and reference intervals.


## Ideas/Blue Sky

use https://github.com/yizhilll/MERT/blob/main/README.md MERT with a chord classification head and fine tune on Billboard/IsoPhonics

You can see an example of genre classifcation here https://huggingface.co/patcasso/mert-classifier

!!!! https://github.com/itamarseg/mert_analysis

Also
ChordSync (2024) — neural audio↔chord alignment (Conformer)
•	Paper (audio-to-chord annotation alignment).  ￼
•	Code repo (they provide a pretrained model + library).  ￼


HUGE https://arxiv.org/html/2507.03482v1

https://github.com/MTG/omar-rq

https://huggingface.co/mtg-upf/omar-rq-multifeature-25hz-fsq

