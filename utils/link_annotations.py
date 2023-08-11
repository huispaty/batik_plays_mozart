'''
This script extracts the harmony, cadence and phrase label annotations from Hentschel et al. ([1], linked as a git submodule) to the score note arrays used in the alignment.

[1] Hentschel, J., Neuwirth, M. and Rohrmeier, M., 2021. The Annotated Mozart Sonatas: Score, Harmony, and Cadence. Transactions of the International Society for Music Information Retrieval, 4(1), p.67–80.DOI: https://doi.org/10.5334/tismir.63
'''

import itertools
import os
import pandas as pd
import numpy as np
import partitura as pt
import xml.etree.ElementTree as ET

# optional imports
import warnings
warnings.filterwarnings('ignore')


# helper functions 
def fraction2float(frac_str):
    # 'quarterbeats' col in annotations.tsv come as fractions, we need float
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        frac = round(float(num) / float(denom), 6)
        return frac

def whole2quarter(frac_str):
    # 'mc_onset' col in annotations.tsv come in whole note unit, we need quarter note unit
    try: # for 0 value
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        frac = round(float(num) / float(denom), 6)
        return frac * 4

def get_num_quarter_per_measure(time_sig):
    # To correct missing 'mc_onset' values (for volta endings), we need to count the num_quarter_notes that fit into a measure given a time sig
    if time_sig.split('/')[1] == '4':
        num_quarter_per_measure = time_sig.split('/')[0]
    elif time_sig.split('/')[1] == '2':
        num_quarter_per_measure = 4
    else: # i.e. 6/8, 3/8 etc
        num_quarter_per_measure = int(time_sig.split('/')[0]) / 2
    return int(num_quarter_per_measure)

# linking functions

def add_missing_duration_qb_in_unfolded_spart_for_volta_first_endings(piece, perf2score_dir):
    piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
    spart_annotated_unfolded = pd.read_csv(os.path.join(piece_dir, 'spart_annotated.csv'))
    # Update missing nan values
    labelled_measures_idxs = spart_annotated_unfolded[~spart_annotated_unfolded.label.isna()].index
    for i, labelled_measure_idx in enumerate(labelled_measures_idxs):
        if not np.isnan(spart_annotated_unfolded.iloc[labelled_measure_idx].volta) and np.isnan(spart_annotated_unfolded.iloc[labelled_measure_idx].duration_qb):
            # next_labels_indxs = [labelled_measures_idxs[i], labelled_measures_idxs[i+1]]
            curr_label_onset_quarter = spart_annotated_unfolded.iloc[labelled_measures_idxs[i]].onset_quarter
            next_label_onset_quarter = spart_annotated_unfolded.iloc[labelled_measures_idxs[i+1]].onset_quarter
            spart_annotated_unfolded.loc[labelled_measure_idx, 'duration_qb'] = next_label_onset_quarter - curr_label_onset_quarter
    spart_annotated_unfolded.to_csv(os.path.join(piece_dir, 'spart_annotated.csv'), index=None)
    return None

def unfold_annotated_spart(piece, annotations, perf2score_dir):
    
    piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
    spart_annotated = pd.read_csv(os.path.join(
        piece_dir, 'spart_annotated_min.csv'))
    spart_unfolded = pd.read_csv(os.path.join(piece_dir, 'spart.csv'))
        
    # Extract labelled note ids
    ids_labelled = spart_annotated.loc[~spart_annotated.label.isna(), 'id':'chord_type']
    # Check that all annotations were linked
    assert np.array_equal(ids_labelled.label.values, annotations.label.values)
    # Merge unfolded spart with labelled spart via note ids
    if spart_unfolded.id.str.contains('-1').any():
        spart_unfolded['tmp'] = spart_unfolded.id.str[:-2]
        spart_annotated_unfolded = spart_unfolded.merge(spart_annotated, left_on='tmp', right_on='id', suffixes=('','_r'))
        spart_annotated_unfolded.drop(columns=['tmp'], inplace=True)
    else:
        spart_annotated_unfolded = spart_unfolded.merge(spart_annotated, on='id', suffixes=('','r'))
    
    # Drop duplicated and non needed columns
    spart_annotated_unfolded.drop(columns=[col for col in spart_annotated_unfolded.columns if '_r' in col], inplace=True)
    spart_annotated_unfolded.drop(columns=['mc', 'mn_onset_quarter', 'quarterbeats', 'mc_onset', 'mn_onset'], inplace=True)
    # Sort spart into original order
    spart_annotated_unfolded = spart_annotated_unfolded.sort_values(by=['onset_beat'])
    
    # Save unfolded labelled spart
    spart_annotated_unfolded.to_csv(os.path.join(piece_dir, 'spart_annotated.csv'), index=None)
    if piece in ['283_2', '331_1', '331_2', '331_3', '333_2', '533_3']:
        add_missing_duration_qb_in_unfolded_spart_for_volta_first_endings(piece, perf2score_dir)
    
    return None

def link_score_note_array_with_annotations(piece, score_dir, annotations_dir, perf2score_dir):
    '''
    Link score note array from MusicXML with the DCML annotations
    '''
    piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
    
    # Create score note array, add xml_mn and mn, save as csv
    score_xml = os.path.join(score_dir, f'kv{piece[:3]}_{piece[-1]}.musicxml')
    score = pt.load_musicxml(score_xml)
    spart = score.parts[0].note_array()
    spart = pd.DataFrame(spart)
    
    measures_notes_dict = {}
    root = ET.parse(score_xml)
    for measure in root.findall("//measure"):
        measure_number = measure.get('number')
        notes = measure.findall('note')
        notes_id = [note.get('id') for note in notes]
        if measure_number in measures_notes_dict:
            measures_notes_dict[measure_number].extend(notes_id)
        else:
            measures_notes_dict[measure_number] = notes_id
    xml_measure_numbers = np.array([key for note_id in spart.id for key,
                                  item in measures_notes_dict.items() if note_id in item])
    spart.insert(0, 'xml_mn', xml_measure_numbers)
    spart.insert(1, 'mn', xml_measure_numbers)
    # Convert partial measures into a measure number that aligns with the counting in mn col of scores and annotations (mn col assumes that volta ending 1 and 2 have same mn)
    if spart['mn'].str.contains('X').any():
        partial_measures_idxs = spart[spart['mn'].str.contains('X')].index 
        partial_measures_idxs = [list(map(lambda x: x[1], g)) for k, g in itertools.groupby(enumerate(partial_measures_idxs), lambda x: x[0]-x[1])]
        for idxs in partial_measures_idxs:
            last_full_mn = spart['xml_mn'].iloc[(min(idxs) - 1)]
            spart['mn'].iloc[idxs] = last_full_mn
    spart['mn'] = spart['mn'].astype('int64') 
    spart.to_csv(os.path.join(piece_dir, 'spart_annotated.csv'), index=None)
    
    # Get annotations and:
    # - drop non-used cols
    # - convert 'quarterbeat' col to float
    # - add missing onset_quarter information for annotations in volta brackets, which in [1] have no onset information
    annotations = pd.read_csv(os.path.join(annotations_dir, f'K{piece[:3]}-{piece[-1]}.tsv'), sep='\t')
    annotations = annotations.drop(annotations.iloc[:,7:9], axis=1) # drop col staff, voice
    annotations = annotations.drop(columns=['pedal','form','figbass', 'changes', 'relativeroot'], axis=1)
    annotations = annotations.drop(annotations.iloc[:,-6:], axis=1) # drop last 6 cols
    # Create onset quarter col from quarterbeats col
    onset_quarter = annotations['quarterbeats'].apply(lambda x: fraction2float(x))
    if spart['onset_quarter'][0] < 0: # offset for anacrusis amount
        onset_quarter += spart['onset_quarter'][0]
    annotations.insert(2, 'onset_quarter', onset_quarter) # index, col_name, values
    # Create mn_onset_quarter col from mn_onset col
    mn_onset_quarter = annotations['mn_onset'].apply(lambda x: whole2quarter(x))
    annotations.insert(3, 'mn_onset_quarter', mn_onset_quarter)
        
    # If the piece contains volta brackets, some mc_onset values are NA. We need to correct that
    if 'volta' in annotations.columns: # ['283_2', '331_1', '331_2', '331_3', '333_2', '533_3']
        onset_quarter_corr = np.zeros(annotations.shape[0])
        
        measure_numbers = annotations.mn # starts at 0 when piece has anacrusis, otherwise at 1
        onset_quarters = annotations.onset_quarter # onset of label in quarter notes
        mn_onset_quarters = annotations.mn_onset_quarter # the distance of the label to the measure start in quarter notes
        time_sigs, voltas = annotations.timesig, annotations.volta
        
        mn_offset = 0 if measure_numbers[0] < 1 else 1  # per default 1, only when pickup 0
        pickup_qu_offset = onset_quarters[0] if onset_quarters[0] < 0 else 0 # the pickup amount offset in quarter notes
        
        volta_measures = []
        volta_mn_offset = 0 # volta endings have the same mn. If we get to the second volta ending, we want to increase the offset.
        
        if len(set(time_sigs)) > 1: # 331_1: multiple volta and a time sig change
            time_sig_change = False
            curr_time_sig = time_sigs[0]
            
            for i, (mn, mn_oqu, time_sig, volta) in enumerate(zip(measure_numbers, mn_onset_quarters, time_sigs, voltas)):
                if volta == 2.0 and mn not in volta_measures:
                    volta_measures.append(mn)
                    volta_mn_offset += 1
                
                if time_sig != curr_time_sig:
                    time_sig_change = True
                    # Get the onset in quarter notes of the measure where the time_sig change occurs
                    # first get onset of last measure that occurs in old time sig
                    last_onset_quarter_old_time_sig = oqu_corr - mn_onset_quarters[i-1]
                    # then compute the should-be first onset
                    first_onset_quarter_new_time_sig = last_onset_quarter_old_time_sig + num_quarter_per_measure
                    first_mn_new_time_sig = mn
                    volta_mn_offset -= 1
                    curr_time_sig = time_sig 
                
                # Compute corrected onset in quarter units as if all onsets (incl. volta endings 1) were counted up
                num_quarter_per_measure = get_num_quarter_per_measure(time_sig)
                if time_sig_change:
                    num_measure_in_new_time_sig = mn - first_mn_new_time_sig
                    oqu_corr = first_onset_quarter_new_time_sig + (num_measure_in_new_time_sig - mn_offset+volta_mn_offset) * \
                    num_quarter_per_measure + mn_oqu + pickup_qu_offset
                else:
                    oqu_corr = (mn-mn_offset+volta_mn_offset) * \
                    num_quarter_per_measure + mn_oqu + pickup_qu_offset
                
                onset_quarter_corr[i] = oqu_corr
        else:
            for i, (mn, mn_oqu, time_sig, volta) in enumerate(zip(measure_numbers, mn_onset_quarters, time_sigs, voltas)):
                if volta == 2.0 and mn not in volta_measures:
                    volta_measures.append(mn)
                    volta_mn_offset += 1
                num_quarter_per_measure = get_num_quarter_per_measure(time_sig)
                oqu_corr = (mn-mn_offset+volta_mn_offset) * \
                    num_quarter_per_measure + mn_oqu + pickup_qu_offset
                onset_quarter_corr[i] = oqu_corr
            
        if spart['onset_quarter'][0] < 0: # offset for anacrusis amount
            onset_quarter_corr -=  1
        annotations['onset_quarter'] = onset_quarter_corr
    
    # Make sure that are no nan onset_quarter values (each label needs an onset position)
    annotations['onset_quarter'] = annotations['onset_quarter'].astype('float32')
    assert not annotations['onset_quarter'].isna().any()
    assert not annotations['onset_quarter'].duplicated().any()
    
    # Merge spart with annotations
    spart_annotated = spart.merge(annotations, on=['mn', 'onset_quarter'], how='left')
    # # Overwrite non-(onset)-unique annotations with NaN via indices
    spart_unique_onset_label_idxs = spart_annotated.groupby(['mn', 'onset_quarter', 'label'], as_index=False).nth(0).index
    # Overwrite duplicated label columns 
    labels_cols = spart_annotated.columns.values[11:]
    spart_annotated.loc[~spart_annotated.index.isin(spart_unique_onset_label_idxs), labels_cols] = np.nan   
    # Check for onsets that were not merged 1:1
    labels_onsets = set(annotations.onset_quarter)
    spart_onsets = set(spart.onset_quarter)
    missing_onsets = list(labels_onsets - spart_onsets)
    if len(missing_onsets) > 0: # [#closest_onset_merge] for ['279_2', '333_3', '533_2'] 
        closest_onset_merge = pd.merge_asof(annotations[annotations['onset_quarter'].isin(
            missing_onsets)], spart, on='onset_quarter', by='mn', direction='nearest', tolerance=.25)
        closest_onset_merge = closest_onset_merge.reindex(
            columns=spart_annotated.columns)
        for i, id in enumerate(closest_onset_merge.id):
            spart_id_idx = spart_annotated.loc[spart_annotated.id == id].index[0]
            spart_annotated.loc[spart_id_idx, labels_cols] = closest_onset_merge.loc[i, labels_cols].values
    
    spart_annotated.to_csv(os.path.join(piece_dir, 'spart_annotated_min.csv'), index=None)
    # Check that no label was omitted
    assert spart.shape[0] - annotations.shape[0] == spart_annotated['label'].isna().sum(), f"{spart.shape[0] - annotations.shape[0], spart_annotated['label'].isna().sum()}"
    
    # Unfold the annotated spart according to the performance structure
    unfold_annotated_spart(piece, annotations, perf2score_dir)
    
    return None


def save_spart_with_annotation_type(piece, perf2score_dir, annotation_type = 'phrases'):
    
    piece_dir = os.path.join(perf2score_dir, f'KV{piece[:3]}', f'kv{piece}')
    spart_annotated = pd.read_csv(os.path.join(piece_dir, 'spart_annotated.csv'))
    
    spart_cols = ['onset_beat', 'duration_beat', 
                  'onset_quarter', 'duration_quarter',
                  'onset_div', 'duration_div', 
                  'pitch', 'voice', 'id']
    
    if annotation_type == 'phrases':
    
        if piece in ['283_2', '331_1', '331_2', '331_3', '333_2', '533_3']:
            phrase_cols =  ['xml_mn', 'mn', 'label', 'duration_qb', 'timesig', 'volta', 'phraseend']
        else:
            phrase_cols =  ['xml_mn', 'mn', 'label', 'duration_qb', 'timesig', 'phraseend']
        label_cols = [phrase_cols[:2], spart_cols, phrase_cols[2:]]
        label_cols = list(itertools.chain(*label_cols))
        spart_phrase_labels = spart_annotated[label_cols]
        
        # # phraseend can have the following values: 
        # # {     -> phrase beginning
        # # }{    -> phrase ending and beginning
        # # }     -> phrase ending. Problem: phrase ending is notated at the same onset point as the harmony label. We need to use duration_qb to move it to the corret onset.
        assert spart_phrase_labels.timesig.isna().sum() == spart_phrase_labels.duration_qb.isna().sum() 
        # overwrite timesig and duration_qb values at onsets were there is no phrase annotation
        non_phrase_label_indices = (~spart_phrase_labels.timesig.isna()) & (spart_phrase_labels.phraseend.isna()) 
        # overwrite timesig and duration_qb values at onsets were there is no phrase annotation    
        spart_phrase_labels.loc[non_phrase_label_indices, ['duration_qb', 'timesig']] = np.nan
        
        # add a new column phrase_type: 1 = {, 2 = }{, 3 = }
        spart_phrase_labels['phrase_type'] = np.zeros(spart_phrase_labels.shape[0], dtype=int)
        spart_phrase_labels.loc[spart_phrase_labels.phraseend =='{', 'phrase_type'] = 1
        spart_phrase_labels.loc[spart_phrase_labels.phraseend =='}{', 'phrase_type'] = 2
        spart_phrase_labels.loc[spart_phrase_labels.phraseend =='}', 'phrase_type'] = 3
        spart_phrase_labels.to_csv(os.path.join(piece_dir, 'spart_phrases.csv'), index=None)
        
    elif annotation_type == 'harmony':
        
        if piece in ['283_2', '331_1', '331_2', '331_3', '333_2', '533_3']:
            phrase_cols = ['xml_mn', 'mn', 'label', 'duration_qb', 'timesig', 'volta', 'globalkey', 'localkey', 'chord', 'numeral', 'chord_type']
        else:
            phrase_cols = ['xml_mn', 'mn', 'label', 'duration_qb', 'timesig', 'globalkey', 'localkey', 'chord', 'numeral', 'chord_type']
        label_cols = [phrase_cols[:2], spart_cols, phrase_cols[2:]]
        label_cols = list(itertools.chain(*label_cols))
        spart_harmony_labels = spart_annotated[label_cols]
        
        spart_harmony_labels['chord_label'] = np.zeros(spart_harmony_labels.shape[0], dtype=np.int64)
        chord_idxs = spart_harmony_labels[~spart_harmony_labels.chord.isna()].index
        spart_harmony_labels.loc[chord_idxs, 'chord_label'] = 1        
        assert spart_harmony_labels.label.isna().sum() == spart_harmony_labels.globalkey.isna().sum() == spart_harmony_labels.localkey.isna().sum() 
        assert spart_harmony_labels.chord.isna().sum() == spart_harmony_labels.numeral.isna().sum() == spart_harmony_labels.chord_type.isna().sum()
    
        spart_harmony_labels.to_csv(os.path.join(piece_dir, 'spart_harmony.csv'), index=None)
    
    elif annotation_type == 'cadence':
    
        if piece in ['283_2', '331_1', '331_2', '331_3', '333_2', '533_3']:
            phrase_cols = ['xml_mn', 'mn', 'label', 'duration_qb', 'timesig', 'volta', 'cadence']
        else:
            phrase_cols = ['xml_mn', 'mn', 'label', 'duration_qb', 'timesig', 'cadence']
        label_cols = [phrase_cols[:2], spart_cols, phrase_cols[2:]]
        label_cols = list(itertools.chain(*label_cols))
        spart_cadence_labels = spart_annotated[label_cols]
        # Add cadence type column:
        # 1 = authentic cadence (PAC and IAC)
        # 2 = half cadence
        # 3 = avoiding cadence (EC and DC)
        spart_cadence_labels['cadence_type'] = np.zeros(spart_cadence_labels.shape[0], dtype=np.int64)
        perfect_cadence_idxs = spart_cadence_labels[(spart_cadence_labels.cadence == 'PAC') |(spart_cadence_labels.cadence == 'IAC')].index
        half_cadence_idxs = spart_cadence_labels[spart_cadence_labels.cadence == 'HC'].index
        avoiding_cadence_idxs = spart_cadence_labels[(spart_cadence_labels.cadence == 'DC') |(spart_cadence_labels.cadence == 'EC')].index
        spart_cadence_labels.loc[perfect_cadence_idxs, 'cadence_type'] = 1
        spart_cadence_labels.loc[half_cadence_idxs, 'cadence_type'] = 2
        spart_cadence_labels.loc[avoiding_cadence_idxs, 'cadence_type'] = 3        
        spart_cadence_labels.to_csv(os.path.join(piece_dir, 'spart_cadence.csv'), index=None)

    return None
