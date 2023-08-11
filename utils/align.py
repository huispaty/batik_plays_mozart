import pandas as pd
import numpy as np
# from .helpers import midi_pitch_to_piano_key

def print_stats(aligned_note_ids, mspart, unaligned_mspart, spart, unaligned_spart, merged_on, merged_by=None, print_aligned_notes=False):
    num_notes_aligned = aligned_note_ids.shape[0]
    num_notes_match = mspart.shape[0]
    num_notes_score = spart.shape[0]
    if merged_by:
        print(f'{num_notes_aligned:4d} from {num_notes_match} ({num_notes_aligned/num_notes_match*100:.2f}%) merged on nearest {merged_on} and exact match on {merged_by}.')
    else:
        print(f'{num_notes_aligned:4d} from {num_notes_match} ({num_notes_aligned/num_notes_match*100:.2f}%) merged on {merged_on}.')
    print(f'{unaligned_mspart.shape[0]:4d} from {num_notes_match} ({unaligned_mspart.shape[0]/num_notes_match*100:.2f}%) match notes left to be aligned.')
    print(f'{unaligned_spart.shape[0]:4d} from {num_notes_match} ({unaligned_spart.shape[0]/num_notes_score*100:.2f}%) score notes left to be aligned.')
    
    if print_aligned_notes:
        print(f'Aligned notes:\n{aligned_note_ids}')
    
    return None

def merge_on_onset_beat_duration_pitch(mspart, spart, align_on=['onset_beat', 'duration_beat', 'pitch']):
    '''
    First-level alignment: Align on same onset and duration (in beat) and pitch
    Accept MergeError due to duplicates: print out duplicates and handle manually
    '''
    try:
        merge = mspart.merge(spart, on=align_on, suffixes=('_m', '_s'), validate='one_to_one')
    except pd.errors.MergeError:
        # if any(spart.duplicated(subset=align_on, keep=False)):
        #     spart_duplicates = spart[spart.duplicated(subset=align_on, keep=False)]
        #     spart_duplicates['pk'] = [midi_pitch_to_piano_key(pitch) for pitch in spart_duplicates['pitch'].values]
        #     # print(f'spart contains {spart_duplicates.shape[0]} notes with shared {align_on}')
        #     # print(spart_duplicates.to_string(index=False))
        # if any(mspart.duplicated(subset=align_on, keep=False)):
        #     mspart_duplicates = mspart[mspart.duplicated(subset=align_on, keep=False)]
        #     mspart_duplicates['pk'] = [midi_pitch_to_piano_key(pitch) for pitch in mspart_duplicates['pitch'].values]
        #     # print(f'mspart contains {mspart_duplicates.shape[0]} notes with shared {align_on}')
        #     # print(mspart_duplicates.to_string(index=False))
        merge = mspart.merge(spart, on=align_on, suffixes=('_m', '_s'))
        
    aligned_note_ids = merge.filter(['id_m', 'id_s'], axis=1)
    unaligned_mspart = mspart[~mspart['id'].isin(merge['id_m'])]
    unaligned_spart = spart[~spart['id'].isin(merge['id_s'])]
    
    # print_stats(aligned_note_ids, mspart, unaligned_mspart, spart, unaligned_spart, merged_on=align_on)

    return aligned_note_ids, unaligned_mspart, unaligned_spart

def merge_on_onset_beat_pitch(mspart, spart, align_on=['onset_beat', 'pitch']):
    '''
    Second-level alignment: Align on same onset (in beat) and pitch
    Accept MergeError due to duplicates: print out duplicates and handle manually
    '''
    try:
        merge = mspart.merge(spart, on=align_on, suffixes=('_m', '_s'), validate='one_to_one')
    except pd.errors.MergeError:
        # if any(spart.duplicated(subset=align_on, keep=False)):
        #     spart_duplicates = spart[spart.duplicated(subset=align_on, keep=False)]
        #     spart_duplicates['pk'] = [midi_pitch_to_piano_key(pitch) for pitch in spart_duplicates['pitch'].values]
        #     # print(f'spart contains {spart_duplicates.shape[0]} notes with shared {align_on}')
        #     # print(spart_duplicates.to_string(index=False))
        # if any(mspart.duplicated(subset=align_on, keep=False)):
        #     mspart_duplicates = mspart[mspart.duplicated(subset=align_on, keep=False)]
        #     mspart_duplicates['pk'] = [midi_pitch_to_piano_key(pitch) for pitch in mspart_duplicates['pitch'].values]
        #     # print(f'mspart contains {mspart_duplicates.shape[0]} notes with shared {align_on}')
        #     # print(mspart_duplicates.to_string(index=False))
        merge = mspart.merge(spart, on=align_on, suffixes=('_m', '_s'))
        # # Remove the second (duplicate) match
        # merge = merge.groupby(['id_m'])['id_s'].first().reset_index()
                
    aligned_note_ids = merge.filter(['id_m', 'id_s'], axis=1)

    unaligned_mspart = mspart[~mspart['id'].isin(merge['id_m'])]
    unaligned_spart = spart[~spart['id'].isin(merge['id_s'])]
    
    # print_stats(aligned_note_ids, mspart, unaligned_mspart, spart, unaligned_spart, merged_on=align_on)

    return aligned_note_ids, unaligned_mspart, unaligned_spart
       
def merge_asof_onset_beat_match_pitch(mspart, spart, on='onset_beat', by='pitch'):    
    '''
    Third-level alignment: Align on closest onset_beat and match by pitch
    '''
    merge_asof = pd.merge_asof(mspart, spart, on=on, by=by, suffixes=('_m', '_s'), tolerance=1, direction='nearest').groupby('onset_beat').first().reset_index()
    # drop NaNs (resulting from allow_exact_matches=True)
    merge_asof.dropna(inplace=True)
    
    aligned_note_ids = merge_asof.filter(['id_m', 'id_s', 'flag'], axis=1)
    unaligned_mspart = mspart[~mspart['id'].isin(merge_asof['id_m'])]
    unaligned_spart = spart[~spart['id'].isin(merge_asof['id_s'])]
    
    # print_stats(aligned_note_ids, mspart, unaligned_mspart, spart, unaligned_spart, merged_on=on, merged_by=by)
    
    return aligned_note_ids, unaligned_mspart, unaligned_spart

def align_mspart_with_spart(mspart, spart):
    # first, merge 1:1 on ['onset_beat', 'duration_beat', 'pitch']
    aligned_note_ids_1, unaligned_mspart, unaligned_spart = merge_on_onset_beat_duration_pitch(mspart, spart)
    # second, merge 1:1 on ['onset_beat', 'pitch']
    aligned_note_ids_2, unaligned_mspart, unaligned_spart = merge_on_onset_beat_pitch(unaligned_mspart, unaligned_spart)
    # third, merge on nearest ['onset_beat'] and match on 'pitch'
    if not unaligned_mspart.empty and not unaligned_spart.empty:
        aligned_note_ids_3, unaligned_mspart, unaligned_spart = merge_asof_onset_beat_match_pitch(unaligned_mspart, unaligned_spart)
    else:
        aligned_note_ids_3 = pd.DataFrame()
    
    aligned_note_ids = pd.concat([aligned_note_ids_1, aligned_note_ids_2, aligned_note_ids_3], ignore_index=True)
    
    # Check if there are unaligned match or score notes
    if unaligned_mspart.empty and unaligned_spart.empty:
        return aligned_note_ids, None, None
    else:
        return aligned_note_ids, unaligned_mspart, unaligned_spart

def check_alignment_completeness(aligned_note_ids, mspart, spart):
        # print('Checking alignment completeness...')
        
        aligned_match_notes = aligned_note_ids['id_m']
        aligned_score_notes = aligned_note_ids['id_s'].str.replace('sv', 's')
        
        # Check if all match and score notes are aligned
        non_aligned_match_ids = mspart['id'][~mspart['id'].isin(aligned_match_notes)]
        non_aligned_score_ids = spart['id'][~spart['id'].isin(aligned_score_notes)]
        
        if non_aligned_match_ids.empty and non_aligned_score_ids.empty:
            # print('All match and score notes aligned')
            return True
            
        else:
            # if not non_aligned_match_ids.empty:
                # print(f'{non_aligned_match_ids.shape[0]} match notes have not been aligned:\n{non_aligned_match_ids}')

            # if not non_aligned_score_ids.empty:
                # print(f'{non_aligned_score_ids.shape[0]} score notes have not been aligned:\n{non_aligned_score_ids}')
            return False

def check_alignment_uniqueness(aligned_note_ids):
            # print('Checking alignment uniqueness...')
            
            aligned_match_notes = aligned_note_ids['id_m']
            aligned_score_notes = aligned_note_ids['id_s'].str.replace('sv', 's')
            
            # Drop 'undefined' values
            if 'undefined' in np.array(aligned_match_notes):
                num_deletions = aligned_match_notes.value_counts()['undefined']
                # print(f'Found {num_deletions} undefined match ids (deletions)')
                aligned_match_notes = aligned_match_notes[aligned_match_notes != 'undefined']
            if 'undefined' in np.array(aligned_score_notes):
                num_insertions = aligned_score_notes.value_counts()['undefined']
                # print(f'Found {num_insertions} undefined score ids (insertions)')
                aligned_score_notes = aligned_score_notes[aligned_score_notes != 'undefined']
        
            # Check if there are match or score notes that are aligned more than once
            if not aligned_match_notes.duplicated().any() and not aligned_score_notes.duplicated().any():
                # print('Each match and score note aligned exactly once')
                return True
            else:
                if aligned_score_notes.duplicated().any():
                    # score_ids_duplicates = np.array(aligned_score_notes.value_counts()[aligned_score_notes.value_counts() > 1].index)
                    score_ids_duplicates = aligned_score_notes[aligned_score_notes.duplicated()]
                    # print(f'Found {len(score_ids_duplicates):2d} doubly aligned score notes: ', score_ids_duplicates.values)
                if aligned_match_notes.duplicated().any():
                    # match_ids_duplicates = np.array(aligned_match_notes.value_counts()[aligned_match_notes.value_counts() > 1].index)
                    match_ids_duplicates = aligned_match_notes[aligned_match_notes.duplicated()]
                    # print(f'Found {len(match_ids_duplicates):2d} doubly aligned match notes: ', match_ids_duplicates.values)
                    
                match_duplicates = aligned_note_ids[aligned_match_notes.duplicated(keep=False)]
                score_duplicates = aligned_note_ids[aligned_score_notes.duplicated(keep=False)]
                if match_duplicates.equals(score_duplicates):
                    aligned_duplicates = match_duplicates
                else:
                    aligned_duplicates = pd.concat([match_duplicates, score_duplicates])
                    aligned_duplicates = aligned_duplicates.drop_duplicates()
                # print('Non-unique alignments: ')
                # print(aligned_duplicates)
                
                return False
