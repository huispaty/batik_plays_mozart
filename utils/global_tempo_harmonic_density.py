'''
Demo 1:
We compare the global tempo (measured in quarter notes per minute) with the label density (in number of labels per minute), to see if there is a correlation. This is a replication of an experiment that was first conducted by Hentschel et al. in [1].

[1] Hentschel, J., Neuwirth, M. and Rohrmeier, M., 2021. The Annotated Mozart Sonatas: Score, Harmony, and Cadence. Transactions of the International Society for Music Information Retrieval, 4(1), p.67–80.DOI: https://doi.org/10.5334/tismir.63
'''

import os
import csv
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import linregress
from .calc_stats import get_corpus_list, get_tempo_indication, get_time_sig

TEMPO_CLASSES_DICT = {'adagio': 0, 'andante': 1, 'allegretto': 2, 'allegro': 3, 'menuetto': 4, 'presto': 5} # for sorting

def extract_tempo_descriptor(score_tempo_indication):
    
    tempo_indications = score_tempo_indication.split(' ')
    for ti in tempo_indications:
        if ti.lower() in list(TEMPO_CLASSES_DICT.keys()):
            return ti.lower()

def create_demo1_data(perf2score_dir, stats_dir, piecewise_quarter_notes_and_labels_csv = 'demo1_piece_quarter_notes_labels.csv'):
    
    demo_data_csv = os.path.join(stats_dir, piecewise_quarter_notes_and_labels_csv)
    
    with open(demo_data_csv, 'w+') as f:
        
        writer = csv.writer(f)
        writer.writerow(['piece', 'tempo', 'num_quarter_notes', 'num_harmony_labels', 'quarter_notes_per_min', 'labels_per_min'])
        
        corpus_pieces_list = get_corpus_list(perf2score_dir)
        
        for piece in corpus_pieces_list:        
            piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
            ppart = pd.read_csv(os.path.join(piece_dir, 'ppart.csv'))

            onset_time_sec = ppart['onset_sec'].iloc[0]
            perf_time_sec = ppart['onset_sec'].iloc[-1] + \
                ppart['duration_sec'].iloc[-1] - onset_time_sec
            perf_time_min = perf_time_sec / 60
            
            spart_unfolded_with_harmony_labels = pd.read_csv(os.path.join(piece_dir, 'spart_harmony.csv'))
            num_harmony_labels = np.count_nonzero(spart_unfolded_with_harmony_labels.chord_label)
            
            last_offset_quarter = np.max(spart_unfolded_with_harmony_labels.onset_quarter) + spart_unfolded_with_harmony_labels.duration_quarter.to_numpy()[-1]
            num_quarter_notes = last_offset_quarter + np.abs(spart_unfolded_with_harmony_labels.onset_quarter[0])
            
            # Per minute 
            labels_per_minute = np.round(num_harmony_labels / perf_time_min, 4)
            quarter_notes_per_minute = np.round(num_quarter_notes / perf_time_min, 4)
            
            mf = os.path.join(piece_dir, f'kv{piece}.match')
            tempo = extract_tempo_descriptor(get_tempo_indication(mf))
            
            writer.writerow([piece, tempo, num_quarter_notes,
                            num_harmony_labels, quarter_notes_per_minute, labels_per_minute])
    
    # Sorting pieces by their tempo class
    piecewise_quarter_notes_and_labels_df = pd.read_csv(demo_data_csv)
    piecewise_quarter_notes_and_labels_df = piecewise_quarter_notes_and_labels_df.sort_values(by=['tempo'], key=lambda x: x.map(TEMPO_CLASSES_DICT))
    piecewise_quarter_notes_and_labels_df.to_csv(demo_data_csv, index=None)
    
    return demo_data_csv


def comp_correlation(demo_data_csv):
    data = pd.read_csv(demo_data_csv)
    x, y = data.quarter_notes_per_min, data.labels_per_min

    lin_reg = linregress(x, y)
    print('Computing correlation between the global tempo and harmonic change rate:')
    print(f"Pearson's r({data.shape[0]}) = {np.round(lin_reg.rvalue, 4)}, p = {lin_reg.pvalue}")
    print(f"slope = {np.round(lin_reg.slope, 4)}")
    return None

def plot_correlation(demo_data_csv, plots_dir, with_piece_labels=False):
    # Plot
    data = pd.read_csv(demo_data_csv)
    labels_per_min, quarter_notes_per_min = data.labels_per_min, data.quarter_notes_per_min
    groups = data.groupby('tempo', sort=False)

    _, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.quarter_notes_per_min, group.labels_per_min, marker='o', linestyle='', ms=4, label=name)
    
    ax.plot(np.unique(quarter_notes_per_min), np.poly1d(np.polyfit(quarter_notes_per_min, labels_per_min, 1))(np.unique(quarter_notes_per_min)), color = 'black', linestyle='dashed')
    ax.legend(frameon=True, loc='upper left')
    ax.set_xlabel('Tempo (quarter notes per minute)')
    ax.set_ylabel('Labels per minute')
    
    if with_piece_labels:
        for _, row in data.iterrows():
            ax.annotate(row['piece'], (row['quarter_notes_per_min'],
                        row['labels_per_min']), fontsize=6)
        plt.savefig(os.path.join(plots_dir, 'demo1_corr_tempo_harmonics_piece_with_labels.png'))    
    else: plt.savefig(os.path.join(plots_dir, 'demo1_corr_tempo_harmonics.png'))



def comp_global_tempo_harmonic_density_correlation(perf2score_dir, stats_dir, plots_dir):
    data = create_demo1_data(perf2score_dir, stats_dir)
    comp_correlation(data)
    plot_correlation(data, plots_dir)
    