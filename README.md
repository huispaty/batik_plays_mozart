# TODOs
- add partitura as dependency in env.yml after release of v1.3.2
- check if env creation works as planned
- run main again and check if all dependencies are correctly installed (numpy<numpy=1.21.2)

--- 
# Batik-plays-Mozart dataset

The [Batik-plays-Mozart dataset](link) is a piano performance dataset containing 12 complete Mozart Piano Sonatas (36 distinct movements) performed on a computer-monitored Bösendorfer grand piano by Viennese concert pianist Roland Batik. The performances are provided in MIDI format (the corresponding audio files are commercially available) and note-levelaligned with scores in the New Mozart Edition in MusicXML and musicological harmony, cadence and phrase annotations previously published in [The Annotated Mozart Sonatas](https://transactions.ismir.net/articles/10.5334/tismir.63). 

This repository contains all performances in MIDI, scores in MusicXML and alignments in [match file](https://arxiv.org/abs/2206.01104) format. The audio files are commercially available. The scores and performances are parsed using [partitura](https://github.com/CPJKU/partitura/), and the alignments are provided in `csv` format.

#### Setup
- Install dependencies:
  - python 3.9
  - partitura 1.3.2
  - numpy 1.21
  - pandas 1.4.1
- If you use conda, you can install the dependencies with: `conda env create -f env.yml`
- To create the dataset, run: `python ./main.py`

The script has been tested in Windows, Linux and Mac OS with python 3.9, and the libraries partitura v1.3.2, numpy v1.21.2 and pandas 1.4.1.

#### Citing
If you use this dataset in any research, please cite the relevant paper:

```
@inproceedings{hu2023batik,
    title = {{The Batik-plays-Mozart Corpus: Linking Performance to Score to Musicological Annotations}},
    author = {Hu, Patricia and Widmer, Gerhard},
    booktitle = {{Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)}},
    year = {2023}
}
```


## Content

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



## Structure

After running `main.py`, you will get the following repository structure:

```
annotations             # the musicological annotations, linked as a submodule
data                    # the input data needed to create the alignments
perf2score              # the performance-score-annotations alignments
plots                   # two analysis plots using the data corpus 
score2score             # the old inferred scores linked to the New Mozart Edition score
stats                   # some statistics on the dataset
utils                   # utility functions to create the dataset
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
