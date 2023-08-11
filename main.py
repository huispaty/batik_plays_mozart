#%% 

import os
import csv
import shutil
import pandas as pd
import numpy as np
import partitura as pt
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from utils import preprocessing
from utils import align
from utils import prepare_new_match
from utils import link_annotations
from utils import calc_stats
from utils import global_tempo_harmonic_density
from utils import local_tempo_cadence_type

'''
Data and Paths
'''
repo_dir = os.path.dirname(os.path.abspath(__file__))

# Data dirs
# Inputs
align_old_dir = os.path.join(repo_dir, 'data', 'alignment_old')
score_dir = os.path.join(repo_dir, 'data', 'score_musicxml')
performance_midi_dir = os.path.join(repo_dir, 'data', 'performance_midi')
score2score_manual_alignments_dir = os.path.join(repo_dir, 'data', 'score2score_manual_alignments')
annotations_dir = os.path.join(repo_dir, 'annotations', 'harmonies')

# Outputs
score2score_dir = os.path.join(repo_dir, 'score2score')
perf2score_dir = os.path.join(repo_dir, 'perf2score')
# Stats
stats_dir = os.path.join(repo_dir, 'stats')
plots_dir = os.path.join(repo_dir, 'plots')

for ddir in [score2score_dir, perf2score_dir, stats_dir, plots_dir]:
    if not os.path.exists(os.path.join(repo_dir, ddir)):
        os.makedirs(os.path.join(repo_dir, ddir))
        print(f'Created dir for {ddir}')

# Corpus
PERF2SCORE_CORPUS = [279, 280, 281, 282, 283, 284, 330, 331, 332, 333, 457, 533]
PERF2SCORE_CORPUS = [str(kv) + '_' + str(suffix) for kv in PERF2SCORE_CORPUS for suffix in range(1, 4)]


'''
Preprocessing
'''
# Create perf2score output dirs with intermediate score/alignment representations
preprocessing.make_score2score_alignment_dirs(repo_dir=repo_dir, dir_name='score2score', pieces=PERF2SCORE_CORPUS)

# Create intermediate score representations using partitura
# unfolded score parts from musicxml
VARIATIONS_PIECES = ['284_3', '331_1', '331_3']
print('Loading unfolded score part with measure numbers from musicxml')
for piece in tqdm(PERF2SCORE_CORPUS):
    if piece in VARIATIONS_PIECES:
        preprocessing.process_unfolded_spart_with_mn_from_score(piece, score_dir, align_old_dir, score2score_dir, create_max_unfolding=True) 
    else:
        preprocessing.process_unfolded_spart_with_mn_from_score(piece, score_dir, align_old_dir, score2score_dir)
print('Finished preprocessing score parts from musicxml files.')
print()

# score parts from match
print('Loading spart with from match...')
for piece in tqdm(PERF2SCORE_CORPUS):
    preprocessing.process_spart_with_mn_from_match(piece, align_old_dir, score2score_dir)
print('Finished preprocessing score parts from old match alignment files.')
print()


'''
Align score2score
'''
for piece in tqdm(PERF2SCORE_CORPUS):
    # Get the spart and mspart used for the alignment
    spart, mspart = preprocessing.get_preprocessed_sparts(piece, score2score_dir)
    # Replace the 'n' in the note ids in score sparts with s
    spart['id'] = spart['id'].str.replace('n', 's')
    
    # Align both sparts
    aligned_note_ids, unaligned_mspart, unaligned_spart = align.align_mspart_with_spart(mspart, spart)
    # save note ids alignment
    path = os.path.join(score2score_dir, f'KV{piece[:3]}', f'kv{piece}', 'aligned_note_ids')
    aligned_note_ids.to_csv(os.path.join(path, 'aligned_note_ids.csv'), index=None) 
    # check if manual corrections were made
    aligned_note_ids_corr = os.path.join(score2score_manual_alignments_dir, f'kv{piece}_score2score_manual.csv')
    if os.path.isfile(aligned_note_ids_corr):
        aligned_note_ids = pd.read_csv(aligned_note_ids_corr)
        aligned_note_ids.to_csv(os.path.join(path, 'aligned_note_ids_manual_corr.csv'), index=None)
    # Check completeness (all notes) and uniqueness (each note aligned once only) in the alignment
    alignment_complete = align.check_alignment_completeness(aligned_note_ids, mspart, spart)
    alignment_unique = align.check_alignment_uniqueness(aligned_note_ids)
    assert alignment_complete and alignment_unique
print(f'Aligned old match score part to new musicxml score part.')
print()

'''
Align perf2score in the matchfile format and link both to the musicological annotations
'''
# Create perf2score dirs
prepare_new_match.make_perf2score_alignment_dirs(repo_dir=repo_dir, dir_name='perf2score', pieces=PERF2SCORE_CORPUS)

for piece in tqdm(PERF2SCORE_CORPUS):
    prepare_new_match.create_matchfile(piece, score2score_dir, perf2score_dir, align_old_dir, score_dir, performance_midi_dir)
    link_annotations.link_score_note_array_with_annotations(piece, score_dir, annotations_dir, perf2score_dir)
    # choose specific annotation type
    link_annotations.save_spart_with_annotation_type(piece, perf2score_dir, annotation_type='phrases')
    link_annotations.save_spart_with_annotation_type(piece, perf2score_dir, annotation_type='harmony')
    link_annotations.save_spart_with_annotation_type(piece, perf2score_dir, annotation_type='cadence')
print(f'All annotations linked for piece {piece}')
print()

'''
Compute some dataset stats
'''
print('Computing overall corpus alignment statistics:')
calc_stats.create_corpus_stats(perf2score_dir, stats_dir)
print()

'''
Do some analysis with the dataset!
'''
# # Here, we analyse how much the global tempo correlates with the harmonic label density
global_tempo_harmonic_density.comp_global_tempo_harmonic_density_correlation(perf2score_dir, stats_dir, plots_dir)
print()

# Next, we analyse
local_tempo_cadence_type.analyse_local_timing_per_cadence_type_and_global_tempo(perf2score_dir, stats_dir, plots_dir)

