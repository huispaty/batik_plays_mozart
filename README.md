# The Batik-plays-Mozart corpus

The [Batik-plays-Mozart corpus](link) is a piano performance-to-score-to-annotations dataset containing 12 complete Mozart Piano Sonatas (36 distinct movements) performed on a computer-monitored Bösendorfer grand piano by Viennese concert pianist Roland Batik. The performances are provided in MIDI format and note-aligned with scores in the New Mozart Edition and musicological harmony, cadence and phrase annotations previously published in [The Annotated 
Mozart Sonatas](https://transactions.ismir.net/articles/10.5334/tismir.63).

This repository contains all performances in MIDI, scores in MusicXML and alignments in [match file](https://arxiv.org/abs/2206.01104) format. The audio files are commercially available.

## Corpus

| sonata | num_pnotes | duration (min) | match (%) | insertion (%) | deletion (%) |
|--------|------------|----------------|-----------|---------------|--------------| 
| KV279  |    7,789   |     16.22      |  94.087   |     5.780     |    0.130     |
| KV280  |    6,277   |     14.72      |  95.793   |     3.983     |    0.223     |
| KV281  |    7,030   |     14.43      |  90.45    |     9.393     |    0.160     |
| KV282  |    5,761   |     14.77      |  96.197   |     3.467     |    0.337     |
| KV283  |    8,231   |     17.42      |  95.657   |     4.233     |    0.107     |
| KV284  |   13,386   |     25.94      |  93.763   |     6.033     |    0.203     |
| KV330  |    7,869   |     18.49      |  96.857   |     3.047     |    0.100     |
| KV331  |   11,760   |     22.66      |  98.283   |     1.370     |    0.347     |
| KV332  |    9,013   |     17.85      |  93.417   |     6.210     |    0.373     |
| KV333  |    9,137   |     20.41      |  96.690   |     3.137     |    0.173     |
| KV457  |    7,290   |     18.25      |  96.043   |     3.843     |    0.110     |
| KV533  |    8,878   |     22.12      |  97.027   |     2.837     |    0.137     |
| **Total**  |  102,421   |    223.28      |  95.355   |     4.444     |    0.2       |




## `main` branch
This branch provides the curated data in the following formats:
```
├── match                       # alignments in match file format
├── midi                        # performance midi files
├── score_parts_annotated       # score parts corresponding to the unfolded performed score structure, aligned to the harmony, cadence and phrase annotations
├── scores                      # score musicxml files
├── annotations                 # the annotations provided by the authors of The Annotated Mozart Sonatas, linked as a submodule
```

## `curate_data` branch
This branch supports linking and 'curating' the data as described in the paper:

### Setup
- Install dependencies:
  - python 3.9
  - partitura 1.4.0
  - numpy 1.21
  - pandas 1.4.1
- If you use conda, you can install the dependencies with: `conda env create -f env.yml`
- Initiate the submodules if this step is not done automatically on cloning: `git submodule init`
- To create the score2score and perf2score2annotations alignments, run: `python ./main.py`


### Structure
After running `main.py`, you will get the following repository structure:
```
annotations             # the musicological annotations, linked as a submodule
data                    # the input data needed to create the alignments
perf2score              # the performance-score-annotations alignments
plots                   # the plots for the two experiments described in the paper
score2score             # the old inferred scores linked to the New Mozart Edition score
stats                   # some statistics on the dataset
utils                   # helper functions for creating the dataset
main.py                 
env.yml                 
```

#### `score2score` subdir structure
Each movement-wise directory is structured as follows:
```
├── KV279
|   ├── kv279_1
|   |   ├── aligned_note_ids        # score note ids aligned
|   |   ├── sparts_from_match       # score note arrays created from old alignment files
|   |   ├── sparts_from_musicxml    # score note arrays created from New Mozart Edition score files
|   |   ├── sparts_preprocessed     # both score note arrays preprocessed
|   ├── kv279_2
|   ├── kv279_3
├── KV280
├── ...
```

#### `perf2score` subdir structure
Each movement-wise directory contains the following files:
```
├── KV279
|   ├── kv279_1
|   |   ├── alignment.csv               # performance-score alignment expressed as aligned note-id pairs
|   |   ├── kv279_1.match               # performance-score alignment in matchfile
|   |   ├── kv279_1.mid                 # performance MIDI
|   |   ├── kv279_1.musicxml            # score musicxml
|   |   ├── ppart.csv                   # the performed part, parsed using partitura
|   |   ├── spart_annotated_min.csv     # the score part without any repeated sections
|   |   ├── spart_annotated.csv         # the unfolded score part with all annotation types
|   |   ├── spart_cadence.csv           # the unfolded score part with cadence annotations
|   |   ├── spart_harmony.csv           # ... harmony annotations
|   |   ├── spart_phrases.csv           # ... phrase annotations
|   |   ├── spart.csv                   # the unfolded score part, parsed using partitura
|   ├── kv279_2
|   ├── kv279_3
├── KV280
├── ...
```

# Citing
If you use this dataset in your research, please cite the relevant paper:

```
@inproceedings{hu2023batik,
    title = {{The Batik-plays-Mozart Corpus: Linking Performance to Score to Musicological Annotations}},
    author = {Hu, Patricia and Widmer, Gerhard},
    booktitle = {{Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)}},
    year = {2023}
}
```

## Acknowledgments
This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (["Whither Music?"](https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/)).
