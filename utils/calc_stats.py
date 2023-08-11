
'''
This script reproduces Table 2 in the accompanying paper.
'''
import re
import os
import pandas as pd
import numpy as np
import partitura as pt
import shutil

# optional
import warnings
warnings.filterwarnings('ignore')

# helpers
def get_corpus_list(perf2score_dir):
    
    corpus_pieces_list = []
    
    for sonata_dir in os.listdir(perf2score_dir):
        sonata_path = os.path.join(perf2score_dir, sonata_dir)
        
        for sonata_mv in os.listdir(sonata_path):
            sonata_mov_path = os.path.join(sonata_path, sonata_mv)
            if os.path.isdir(sonata_mov_path) and sonata_mv.startswith('kv'):
                corpus_pieces_list.append(sonata_mv[2:])
    
    return (sorted(corpus_pieces_list))
    
def get_tempo_indication(mf):
    
    content = None
    with open(mf, 'r') as mfile:
        for line in mfile:
            match = re.search(r'scoreprop\(tempoIndication,(.*?),', line)
            if match:
                content = match.group(1)
    
    return content

def get_time_sig(piece, perf2score_dir):
    piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
    spart_labelled = pd.read_csv(os.path.join(piece_dir, 'spart_annotated.csv'))
    time_sig = np.unique(spart_labelled[~spart_labelled.timesig.isna()]['timesig'])

    if len(time_sig) > 1:
        return list(time_sig)
    else: return time_sig[0]
        
def number_to_roman(number):
    if number == 1:
        return 'i'
    elif number == 2:
        return 'ii'
    else: return 'iii'


def get_matchfile_stats(matchfile):
    
    match_pattern = r'snote\((.*?)\)-note\((.*?)\).'
    ins_pattern = r'insertion-note\((.*?)\).'
    del_pattern = r'snote\((.*?)\)-deletion.'

    with open(matchfile, 'r') as m_file:
        match_data = m_file.read()
        matches = re.findall(match_pattern, match_data)
        num_matches = len(matches)
        insertions = re.findall(ins_pattern, match_data)
        num_insertions = len(insertions)
        deletions = re.findall(del_pattern, match_data)
        num_deletions = len(deletions)
    num_notes_total = num_matches + num_insertions + num_deletions
    
    return num_matches, num_insertions, num_deletions, num_notes_total



def get_performed_notes_and_performed_time(piece, perf2score_piece_dir):
    
    ppart = pd.read_csv(os.path.join(perf2score_piece_dir, f'ppart.csv'))
    num_pnotes = ppart.shape[0]
    
    onset_time_sec = ppart['onset_sec'].iloc[0]
    perf_time_sec = ppart['onset_sec'].iloc[-1] + \
        ppart['duration_sec'].iloc[-1] - onset_time_sec
    perf_time_min = perf_time_sec / 60
    
    return num_pnotes, np.round(perf_time_min, 2)

def get_num_score_notes(piece, perf2score_piece_dir):
    
    spart = pd.read_csv(os.path.join(perf2score_piece_dir, f'spart.csv'))
    
    snote_ids = spart['id'].tolist()
    num_snotes = len(snote_ids)
    num_unique_snotes = len(set([snote_id.split('-')[0] for snote_id in snote_ids]))
    
    return num_snotes, num_unique_snotes


'''
Calc stats for the corpus
'''
def create_corpus_stats(perf2score_dir, stats_dir, latex_thousands_formatting=False):
    corpus_pieces_list = get_corpus_list(perf2score_dir)
    
    # Stats for every individual movement
    # for pretty printing
    sonata_list = [f'KV{piece[:3]}' for piece in corpus_pieces_list]
    movement_list = [number_to_roman(int(piece[-1])) for piece in corpus_pieces_list]

    tempo_indication_list = []
    time_sigs_list = []
    num_matches_abs = []
    num_matches_rel = []
    num_insertions_abs = []
    num_insertions_rel = []
    num_deletions_abs = []
    num_deletions_rel = []
    num_pnotes_list = []
    duration_min_list = []
    num_snotes_list = []
    num_unique_snotes_list = []

    for piece in corpus_pieces_list:
        piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
        
        matchfile = os.path.join(piece_dir, f'kv{piece}.match')
        tempo = get_tempo_indication(matchfile)
        tempo_indication_list.append(tempo)
        # time sig is read from labels
        time_sig = get_time_sig(piece, perf2score_dir)
        time_sigs_list.append(time_sig)
        # Get alignment stats from new match files
        num_matches, num_insertions, num_deletions, num_notes_total = get_matchfile_stats(matchfile)
        num_matches_abs.append(num_matches)
        num_matches_rel.append(np.round(num_matches / num_notes_total * 100, 2))
        num_insertions_abs.append(num_insertions)
        num_insertions_rel.append(np.round(num_insertions / num_notes_total * 100, 2))
        num_deletions_abs.append(num_deletions)
        num_deletions_rel.append(np.round(num_deletions / num_notes_total * 100, 2))
        # Get performed time in minutes
        num_pnotes, duration_min = get_performed_notes_and_performed_time(piece, piece_dir)
        num_pnotes_list.append(num_pnotes)
        duration_min_list.append(duration_min)
        # Get score stats (num notes vs unique)
        num_snotes, num_unique_snotes = get_num_score_notes(piece, piece_dir)
        num_snotes_list.append(num_snotes)
        num_unique_snotes_list.append(num_unique_snotes)
    
    # Writing it all to a DataFrame
    print(42 * '=')
    corpus_df = pd.DataFrame()
    corpus_df['sonata'] = sonata_list
    corpus_df['mvmt'] = movement_list
    corpus_df['tempo'] = tempo_indication_list
    corpus_df['time_sig'] = time_sigs_list
    corpus_df['num_pnotes'] = num_pnotes_list
    corpus_df['duration (min)'] = duration_min_list
    corpus_df['match_abs'] = num_matches_abs
    corpus_df['match_rel'] = num_matches_rel
    corpus_df['inserts_abs'] = num_insertions_abs
    corpus_df['inserts_rel'] = num_insertions_rel
    corpus_df['del_abs'] = num_deletions_abs
    corpus_df['del_rel'] = num_deletions_rel
    
    header = "{:<10s} {:<6s} {:<6s} {:<6s}"
    data_format = "{:<10s} {:>6.2f} {:>6.2f} {:>6.2f}"
    print(header.format("Label", "Mean", "Min", "Max"))
    print(data_format.format("Match", corpus_df['match_rel'].aggregate('mean'), corpus_df['match_rel'].aggregate('min'), corpus_df['match_rel'].aggregate('max')))
    print(data_format.format("Insert", corpus_df['inserts_rel'].aggregate('mean'), corpus_df['inserts_rel'].aggregate('min'), corpus_df['inserts_rel'].aggregate('max')))
    print(data_format.format("Del", corpus_df['del_rel'].aggregate('mean'), corpus_df['del_rel'].aggregate('min'), corpus_df['del_rel'].aggregate('max')))
    print(42 * '=')
    
    corpus_df.to_csv(os.path.join(stats_dir, 'corpus_movements.csv'), index=None)
    '''
    corpus_movements.csv
    sonata | mvmt | tempo | time_sig | duration (min) | num_snotes | num_snotes_unique | match_abs | match_rel | inserts_abs | inserts_rel | del_abs | del_rel
    '''
    
    # Aggregate for each sonata
    corpus_aggregate_df = corpus_df.groupby('sonata', as_index=False).aggregate({
                            'num_pnotes':'sum', 
                            'duration (min)':'sum', 
                            'match_abs':'sum', 
                            'match_rel':'mean', 
                            'inserts_abs':'sum', 
                            'inserts_rel':'mean', 
                            'del_abs':'sum', 
                            'del_rel':'mean', 
                            })

    corpus_aggregate_row = corpus_aggregate_df.aggregate({
                            'num_pnotes':'sum', 
                            'duration (min)':'sum', 
                            'match_abs':'sum', 
                            'match_rel':'mean', 
                            'inserts_abs':'sum', 
                            'inserts_rel':'mean', 
                            'del_abs':'sum', 
                            'del_rel':'mean', 
                            })
    corpus_aggregate_row['sonata'] = 'Total'
    corpus_aggregate_row.name = 'Total'
    corpus_aggregate_df = corpus_aggregate_df.append(corpus_aggregate_row)
    corpus_aggregate_df['num_pnotes'] = corpus_aggregate_df['num_pnotes'].astype(int)
    corpus_aggregate_df['duration (min)'] = corpus_aggregate_df['duration (min)'].apply(
        lambda x: round(x, 3))
    corpus_aggregate_df['match_abs'] = corpus_aggregate_df['match_abs'].astype(int)
    corpus_aggregate_df['inserts_abs'] = corpus_aggregate_df['inserts_abs'].astype(int)
    corpus_aggregate_df['del_abs'] = corpus_aggregate_df['del_abs'].astype(int)
    # corpus_aggregate_df[['num_snotes', 'num_snotes_unique']] = corpus_aggregate_df[['num_snotes', 'num_snotes_unique']].astype(int)
    corpus_aggregate_df.loc[:, 'match_abs':'del_rel'] = corpus_aggregate_df.loc[:, 'match_abs':'del_rel'].apply(lambda x: round(x, 3))
    corpus_aggregate_df.to_csv(os.path.join(stats_dir, 'corpus_sonatas.csv'), index=None)
    '''
    corpus_sonatas.csv
    sonata | num_pnotes | duration (min) | match_abs | match_rel | inserts_abs | inserts_rel | del_abs | del_rel
    '''

    # Save as latex
    corpus_aggregate_table = corpus_aggregate_df.copy()
    if latex_thousands_formatting == True:
        corpus_aggregate_table['num_pnotes'] = corpus_aggregate_table['num_pnotes'].apply('{:,d}'.format)
        corpus_aggregate_table['match_abs'] = corpus_aggregate_table['match_abs'].apply('{:,d}'.format)
        corpus_aggregate_table['inserts_abs'] = corpus_aggregate_table['inserts_abs'].apply('{:,d}'.format)
        corpus_aggregate_table['del_abs'] = corpus_aggregate_table['del_abs'].apply(
            '{:,d}'.format)
        corpus_aggregate_table.to_latex(os.path.join(stats_dir, 'table2_corpus_sonatas_formatted.tex'), index=None)
    else:
        corpus_aggregate_table.to_latex(os.path.join(stats_dir, 'table2_corpus_sonatas.tex'), index=None)
