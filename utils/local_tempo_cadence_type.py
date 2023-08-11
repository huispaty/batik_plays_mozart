'''
Exp 2:
Here we compare the local tempo evolution before and after a cadence label by using a window of one quarter note prior and subsequent to a sequence label. We use the beat period (IOI_perf_{t, t+1} / IOI_score_{t, t+1}) as a proxy for tempo, and compute tempo values at each score onset where there is a corresponding performed onset.
To get tempo values at regular time steps within the window, we perform time-wise interpolation. Then we plot the resulting curves (their mean and 95% confidence interval) for cadences falling on a downbeat and those falling on a weak beat, and for each type, differentiate between (1) authentic and half cadences, (2) tempo curves at different global tempi.
'''

import os
import csv

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import partitura as pt
from tqdm import tqdm

from .calc_stats import get_corpus_list

# helper functions
def load_alignment(alignment_csv):
    with open(alignment_csv) as f:
        alignment = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]
    return alignment

def get_cadence_labels(cadence_labels, cadence_type='authentic'):
    label_filter = 1 if cadence_type == 'authentic' else (2 if cadence_type == 'half' else 3)
    
    cadence_idxs = np.array(cadence_labels[cadence_labels.cadence_type == label_filter].index)
    cadence_onset_quarter = np.array(cadence_labels[cadence_labels.cadence_type == label_filter].onset_quarter)
    
    return cadence_labels, cadence_idxs, cadence_onset_quarter

def get_snote_array_and_performance_curves(piece, piece_dir):
    # Load unfolded (performed) score and get unique onset indices
    score = pt.load_score(os.path.join(piece_dir, f"kv{piece[:3]}_{piece[-1]}.musicxml"))
    spart = score.parts[0]
    if piece not in ['284_3', '331_1', '331_3']: # variation pieces, take max unfolding 
        alignment = load_alignment(os.path.join(piece_dir, 'alignment.csv'))
        spart = pt.score.unfold_part_alignment(spart, alignment)
    else:
        spart = pt.score.unfold_part_maximal(spart, update_ids=True)
    snote_array = spart.note_array(include_grace_notes=True, include_metrical_position=True)
    # Filter out grace notes
    snote_array = snote_array[np.where(snote_array['is_grace'] != 1)[0]]
    # Get unique score onsets and their indices
    unique_score_onsets_idxs, unique_score_onsets = pt.musicanalysis.performance_codec.get_unique_onset_idxs(snote_array["onset_quarter"], return_unique_onsets=True)
    num_score_onsets = len(unique_score_onsets_idxs)
    # Create a mask indicating if an onset is a downbeat
    note_is_downbeat = snote_array['is_downbeat']
    onset_is_downbeat_mask = np.zeros(len(unique_score_onsets_idxs), dtype=int)
    for i, indices in enumerate(unique_score_onsets_idxs):
        if np.any(note_is_downbeat[indices] == 1):
            onset_is_downbeat_mask[i] = 1
        
    # Load matchfile
    match_file = os.path.join(piece_dir, f"kv{piece[:3]}_{piece[-1]}.match")
    perf, alignment = pt.load_match(match_file)
    ppart = perf.performedparts[0]
    pnote_array = ppart.note_array()
    
    # Get performance array [nd_array of shape (num_notes, )]
    performance_array_notewise, _ = pt.musicanalysis.encode_performance(spart, ppart, alignment)
    # Get indices of matched notes from alignmend
    match_idxs = pt.musicanalysis.performance_codec.get_matched_notes(snote_array, pnote_array, alignment)
    # Filter the part of the note array that was actually performed
    performed_snote_array = snote_array[match_idxs[:, 0]]
    # Get unique performed score onsets
    unique_performed_score_onset_idxs, unique_performed_score_onsets = pt.musicanalysis.performance_codec.get_unique_onset_idxs(performed_snote_array["onset_quarter"], return_unique_onsets=True)
    num_performed_score_onsets = len(unique_performed_score_onset_idxs)
    
    # Check which notated score onsets are missing in the performance
    if num_score_onsets != num_performed_score_onsets:
        # # We want to get an onsetwise performance array
        unique_performed_score_onset_idxs_mask = unique_score_onsets_idxs.copy()
        for score_onset_idx, score_onset in enumerate(unique_score_onsets):
            performed_onset_idx, = np.where(np.isclose(unique_performed_score_onsets, score_onset))
            if performed_onset_idx.size == 0:  # score onset is not found in performed score onset
                performed_onset_idx = last_poi
                if score_onset > np.max(unique_performed_score_onsets):
                    # missing onset occurs at end of piece: take values from last performed onset
                    unique_performed_score_onset_idxs_mask[score_onset_idx] = unique_performed_score_onset_idxs[-1]
                else:
                    # missing onset occurs in the middle: take the mean of the last and next performed onset
                    unique_performed_score_onset_idxs_mask[score_onset_idx] = np.hstack((unique_performed_score_onset_idxs[performed_onset_idx], unique_performed_score_onset_idxs[performed_onset_idx+1]))
            else:
                unique_performed_score_onset_idxs_mask[score_onset_idx] = unique_performed_score_onset_idxs[performed_onset_idx[0]]
                # Save the last matched performance index
                last_poi = performed_onset_idx[0]
        unique_performed_score_onset_idxs = unique_performed_score_onset_idxs_mask
    
    # Get onsetwise performance
    performance_array_onsetwise = pt.musicanalysis.performance_codec.notewise_to_onsetwise(performance_array_notewise,  unique_performed_score_onset_idxs)
    performance_curves = structured_to_unstructured(performance_array_onsetwise)
    
    return snote_array, onset_is_downbeat_mask, performance_curves


'''
Prep data for experiment 2:

demo2_piecewise_aggregate_cadence_data.csv:
piece | tempo | time_sig | authentic | ac_% | half | hc_% | authentic_downbeat | ad_% | half_downbeat | hd_%
-- each row expresses aggregate information for that specific piece
-- pieces are sorted according to their tempo class

demo2_tempo_curves.csv:
piece | tempo | cadence_type | beat_level | measure | score_onset | beat_period | interp
-- each row corresponds to the tempo information at a time step (either real or interpolated, when interpolated, interp=1)
''' 

TEMPO_CLASSES_DICT = {'adagio': 0, 'andante': 1, 'allegretto': 2, 'allegro': 3, 'menuetto': 4, 'presto': 5}

def create_demo2_data(perf2score_dir, stats_dir):
    
    # Prep data files
    piecewise_aggregated_cadence_data_csv = os.path.join(stats_dir, 'demo2_piecewise_aggregate_cadence_data.csv')
    tempo_curves_csv = os.path.join(stats_dir, 'demo2_tempo_curves.csv')
    
    # Refactor from first demo data: the list of pieces and their tempo descriptor
    piecewise_aggregates_df = pd.read_csv(os.path.join(stats_dir, 'demo1_piece_quarter_notes_labels.csv')).loc[:,'piece':'tempo']
    demo2_aggregates_columns = ['authentic', 'a_%', 'half', 'h_%', 'authentic_downbeat', 'ad_%', 'half_downbeat', 'hd_%']
    for col in demo2_aggregates_columns:
        piecewise_aggregates_df[col] = ''
    piecewise_aggregates_df.to_csv(piecewise_aggregated_cadence_data_csv, index=None)
    
    with open(tempo_curves_csv, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['piece', 'tempo', 'cadence_type', 'beat_level', 'measure', 'score_onset', 'beat_period', 'interp'])

    cadence_types = ['authentic', 'half'] # authentic and half make up > 90% of cadences

    corpus_pieces_list = get_corpus_list(perf2score_dir)
    
    print('Computing tempo curves...')
    for piece in tqdm(corpus_pieces_list):
            piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
            # Get score note array and performance curves, where grace notes are filtered out
            snote_array, onset_is_downbeat_mask, onsetwise_performance_curves = get_snote_array_and_performance_curves(piece, piece_dir)
            unique_score_onset_idxs, unique_score_onsets = pt.musicanalysis.performance_codec.get_unique_onset_idxs(snote_array["onset_quarter"], return_unique_onsets=True)
            assert len(unique_score_onset_idxs) == len(unique_score_onset_idxs) == onsetwise_performance_curves.shape[0]
            
            with open(tempo_curves_csv, 'ab') as f:
                for cadence_type in cadence_types:
                    # Get cadence labels
                    cadence_annotations = pd.read_csv(os.path.join(piece_dir, 'spart_cadence.csv'))
                    cadence_labels, cadence_idxs, cadence_onset_quarter = get_cadence_labels(cadence_annotations, cadence_type=cadence_type)
                    
                    num_cadences = 0
                    downbeat_cadences = 0
                    
                    for cadence_idx, cadence_onset_quarter in zip(cadence_idxs, cadence_onset_quarter):
                        measure = cadence_labels.iloc[cadence_idx].xml_mn
                        # Take window of 1 whole note up to cad label + 1 quarter note post label
                        onset_range_lower_limit = cadence_onset_quarter - 1
                        onset_range_upper_limit = cadence_onset_quarter + 1

                        # Get the corresponding score onsets and tempo curve values
                        score_onset_quarter_idx = np.where(np.logical_and(
                            unique_score_onsets >= onset_range_lower_limit-2, unique_score_onsets <= onset_range_upper_limit))
                        score_onsets_in_window = unique_score_onsets[score_onset_quarter_idx[0]]
                        performance_curve_in_window = onsetwise_performance_curves[score_onset_quarter_idx[0]]
                        tempo_curve_in_window = performance_curve_in_window[:, 0]
                    
                        # Check if the cadence falls on a downbeat
                        cad_score_onset_idx = np.where(unique_score_onsets == cadence_onset_quarter)[0]
                        cad_on_downbeat = onset_is_downbeat_mask[cad_score_onset_idx][0]
                        
                        beat_level = 'downbeat' if cad_on_downbeat else 'weak beat'
                        
                        # Do time wise interpolation
                        ## Find missing time steps
                        num_steps = 5 # 2 quarter beats window, with 8th note time steps
                        note_onsets_in_window = np.linspace(onset_range_lower_limit, onset_range_upper_limit, num_steps)
                        missing_onsets = list(set(note_onsets_in_window) - set(score_onsets_in_window))
                        ## Check how many values are interpolated
                        interp_mask = np.zeros(len(note_onsets_in_window), dtype=int)
                        interp_indices = np.where(np.in1d(note_onsets_in_window, missing_onsets))[0]
                        interp_mask[interp_indices] = 1
                        
                        # Filter out curves where the tempo value at the penultimate time point before the cadence label is interpolated
                        if interp_mask[1] == 0:
                            num_cadences += 1
                            if cad_on_downbeat:
                                downbeat_cadences += 1
                            ## Create a df consisting of the score onsets and the beat period at those onsets
                            observed_df = pd.DataFrame(
                                {'score_onset': score_onsets_in_window, 
                                'beat_period': tempo_curve_in_window})
                            ## Create another df with the missing score onsets
                            missing_onsets_df = pd.DataFrame({'score_onset': missing_onsets, 'beat_period': np.repeat(np.nan, len(missing_onsets))})
                            ## Combine the two to do time wise interpolation
                            interpolated_df = pd.concat([observed_df, missing_onsets_df]).sort_values('score_onset').reset_index(drop=True)
                            interpolated_df['score_time'] = pd.to_datetime(interpolated_df['score_onset'], unit='m')
                            interpolated_df = interpolated_df.set_index('score_time')
                            interpolated_df = interpolated_df.interpolate(method='time')
                            ## Filter out only time points we want to use for plotting
                            interpolated_df = interpolated_df[interpolated_df['score_onset'].isin(note_onsets_in_window)]
                            ## Reset the score_onset to have uniform time steps
                            interpolated_df['score_onset'] = interpolated_df['score_onset'] - cadence_onset_quarter
                            ## Reset the index
                            interpolated_df = interpolated_df.reset_index(drop=True)
                            
                            # Add information on the piece, tempo label, cad type and beat level
                            tempo_indication = piecewise_aggregates_df[piecewise_aggregates_df.piece == piece].tempo
                            tempo = np.array(np.repeat(tempo_indication, num_steps))
                            interpolated_df.insert(0, 'piece', piece)
                            interpolated_df.insert(1, 'tempo', tempo)
                            interpolated_df.insert(2, 'cadence_type', cadence_type)
                            interpolated_df.insert(3, 'beat_level', beat_level)
                            interpolated_df.insert(4, 'measure', measure)
                            interpolated_df.insert(7, 'interp', interp_mask)
                            # Save data
                            np.savetxt(f, interpolated_df.to_numpy(), fmt='%s', delimiter=",")
                    
                    # Save aggregate information
                    piecewise_aggregates_df.loc[piecewise_aggregates_df.piece==piece, f'{cadence_type}'] = num_cadences
                    piecewise_aggregates_df.loc[piecewise_aggregates_df.piece==piece, f'{cadence_type}_downbeat'] = downbeat_cadences
    
    piecewise_aggregates_df.to_csv(piecewise_aggregated_cadence_data_csv, index=None)
    
    # Fill in missing values for aggregate information
    piecewise_aggregates_df['authentic'] = piecewise_aggregates_df['authentic'].astype(int)
    piecewise_aggregates_df['half'] = piecewise_aggregates_df['half'].astype(int)
    piecewise_aggregates_df['authentic_downbeat'] = piecewise_aggregates_df['authentic_downbeat'].astype(int)
    piecewise_aggregates_df['half_downbeat'] = piecewise_aggregates_df['half_downbeat'].astype(int)
    piecewise_aggregates_df['a_%'] = (piecewise_aggregates_df['authentic'] / \
        (piecewise_aggregates_df['authentic'] + piecewise_aggregates_df['half']) * 100).astype(float)
    piecewise_aggregates_df['h_%'] = (piecewise_aggregates_df['half'] / \
        (piecewise_aggregates_df['authentic'] + piecewise_aggregates_df['half']) * 100).astype(float)
    piecewise_aggregates_df['ad_%'] = (piecewise_aggregates_df['authentic_downbeat'] / piecewise_aggregates_df['authentic'] * 100).astype(float)
    piecewise_aggregates_df['hd_%'] = (piecewise_aggregates_df['half_downbeat'] / piecewise_aggregates_df['half'] * 100).astype(float)
    piecewise_aggregates_df = piecewise_aggregates_df.round(decimals=2)
    piecewise_aggregates_df = piecewise_aggregates_df.sort_values(
        by=['tempo'], key=lambda x: x.map(TEMPO_CLASSES_DICT))
    piecewise_aggregates_df.to_csv(piecewise_aggregated_cadence_data_csv, index=None)
    
    return piecewise_aggregated_cadence_data_csv, tempo_curves_csv   

def plot_tempo_curves(tempo_curves_csv, plots_dir, save_plot=True):
    curves = pd.read_csv(tempo_curves_csv)
    
    grid = sns.relplot(
    data=curves, x="score_onset", y="beat_period",
    col="beat_level",
    hue="cadence_type",
    style="tempo", style_order=['adagio', 'andante', 'allegretto', 'allegro', 'menuetto', 'presto'],
    kind="line",
    height=3, aspect=1
    )

    grid.axes[0, 0].axvline(0, c='g')
    grid.axes[0, 1].axvline(0, c='g')
    label = {'cadence_onset': plt.axvline(0, c='g')}
    grid.axes[0, 0].legend(label.values(), label.keys(), loc='upper left')
    grid.axes[0, 1].legend(label.values(), label.keys(), loc='upper left')
    sns.move_legend(grid, "upper left", bbox_to_anchor=(.84, .9))
    
    if save_plot:
        plt.savefig(os.path.join(plots_dir, 'demo2_tempo_cadence_beat_level_comparison'), dpi=1200, bbox_inches='tight')
    else:
        plt.show()


def comp_tempo_curves_stats(tempo_curves_csv):
    num_steps = 5
    curves_df = pd.read_csv(tempo_curves_csv)
    num_tempo_values = curves_df.shape[0]
    num_interp = curves_df.interp.sum()
    # Convert to numpy and reshape
    curves_data = curves_df.to_numpy().reshape(-1, num_steps, curves_df.shape[1])
    # print(f'Computed {curves_data.shape[0]} curves from {num_tempo_values:,} tempo values, from which {num_interp:,} ({num_interp/num_tempo_values*100:.2f}%) are timewise interpolated')
    print(f'Plotted {curves_data.shape[0]} curves, split into {len(TEMPO_CLASSES_DICT.keys())} tempo classes.')
    
    # interp_arrays = curves_data[:, :, -1]
    # interp_distr = np.sum(interp_arrays, axis=0)
    # time_steps = np.linspace(-1, 1, num_steps)
    # print(f'Time steps: {time_steps}')
    # print(f'Interp abs: {interp_distr}')
    # rel_distr = np.array(interp_distr/curves_data.shape[0]*100).astype(float)
    # print(f'Interp   %: {np.round(rel_distr, 2)}')
    return None


def analyse_local_timing_per_cadence_type_and_global_tempo(perf2score_dir, stats_dir, plots_dir):
    piecewise_aggregated_cadence_data_csv, tempo_curves_csv = create_demo2_data(perf2score_dir, stats_dir)
    plot_tempo_curves(tempo_curves_csv, plots_dir)
    comp_tempo_curves_stats(tempo_curves_csv)


    