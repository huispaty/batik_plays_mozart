import os
import pandas as pd
import numpy as np
import partitura as pt
import xml.etree.ElementTree as ET
import random

def make_score2score_alignment_dirs(repo_dir, dir_name, pieces):
    aligned_pieces_dir = os.path.join(repo_dir, dir_name)
    if not os.path.exists(aligned_pieces_dir):
        os.mkdir(aligned_pieces_dir)
    
    for p in pieces:
        # Create the necessary dirs
        sonata_dir = os.path.join(aligned_pieces_dir, f'KV{p[:3]}')
        if not os.path.exists(sonata_dir):
            os.mkdir(os.path.join(sonata_dir))
            movements = [f'kv{p[:3]}_{mv}' for mv in range(1, 4)]
            
            prealign_working_dirs = ['sparts_from_match', 'sparts_from_musicxml', 'sparts_preprocessed', 'aligned_note_ids']
            for mv in movements:
                os.mkdir(os.path.join(sonata_dir, mv))
                for wdir in prealign_working_dirs:
                    os.mkdir(os.path.join(sonata_dir, mv, wdir))
        else:
            pass
    
    test_piece = random.choice(pieces)
    test_piece_path = os.path.join(aligned_pieces_dir, f'KV{test_piece[:3]}', f'kv{test_piece}')
    assert os.path.exists(test_piece_path)
        
# Preprocessing from score files
def create_measures_notes_dict(score_path, piece):
    """Function that creates a {'measure_number': [note_1, note_2, note_3]} dict listing which notes belong to which measure in a musicxml score file

    Parameters
    ----------
        score_path : path
            path to musicxml score file
        piece : str
            the name of the piece. For the Batik/Mozart corpus, in the form: kv#_mv#

    Returns
    -------
        measures_notes_dict [dict]: [The {measures: [note1, note2, note3..]} dictionary for a given musicxml file]
    """
    measures_notes_dict = {}
    musicxml_file = os.path.join(score_path, f'kv{piece}.musicxml')
    root = ET.parse(musicxml_file)
    
    for measure in root.findall(".//measure"):
        measure_number = measure.get('number')
        notes = measure.findall('note')
        notes_id = [note.get('id') for note in notes]
        # measures can be organised within parts, or parts within measures.
        # handles the case where the same measures are split between different parts
        if measure_number in measures_notes_dict:
            measures_notes_dict[measure_number].extend(notes_id)
        else:
            measures_notes_dict[measure_number] = notes_id
    
    return measures_notes_dict

def create_unfolded_score_spart_with_mn(score_path, match_path, piece, unfold_structure='performed'):
    """Find the score unfolding variant that corresponds to a performance in a match file.
    
    Create the corresponding spart.note_array() and add measure numbers induced via beat map or found in mxml.

    Parameters
    ----------
    score_path : path
        path to musicxml score file
    match_path : path: 
        path to match file
    piece : str
        name of the piece. For the Batik/Mozart corpus, in the form: kv#_mv#
    unfold_structure : str
        unfolding structure to be used for creating the part. If 'performed': all score variants are compared and the closest (diff. num_notes) is chosen. If 'max': the max score variant is chosen. This is used for creating the score parts of variation pieces

    Returns
    -------
    pd.DataFrame
        The spart note array in the form of a pd.Df including induced measure numbers

    """
    # musicxml_file = os.path.join(score_path, 'K%s-%s.musicxml' %(piece[:3], piece[-1]))
    musicxml_file = os.path.join(score_path, f'kv{piece}.musicxml')
    score = pt.load_musicxml(musicxml_file)
    
    match_file = os.path.join(match_path, 'kv%s.match' %piece)
    _, _, match_part = pt.io.importmatch.load_match(match_file, create_part=True)
    match_part_length = match_part.note_array().shape[0]
    
    if unfold_structure == 'performed':  
        if len(score.parts) > 1: # fix for piece 332_2
            score_part = pt.score.merge_parts(score.parts)
        else:
            score_part = score.parts[0]
        score_variants = pt.score.make_score_variants(score_part)
        score_part_variants_lengths = [score_variant.create_variant_part().note_array().shape[0] for score_variant in score_variants]

        best_match = min(score_part_variants_lengths,
                            key=lambda x: abs(x - match_part_length))
        best_match_idx = score_part_variants_lengths.index(best_match)
        score_part = score_variants[best_match_idx].create_variant_part()
        # Update note IDs to reflect repetition structure (inplace method)
        pt.utils.update_note_ids_after_unfolding(score_part)
    elif unfold_structure == 'max':
        score_part = pt.score.unfold_part_maximal(score[0], update_ids=True)
    
    # Create a spart note array in the form a pd.DataFrame
    score_part_df = pd.DataFrame(score_part.note_array())

    # Add measure numbers
    # Find measure numbers via beat_map
    try: 
        measures_via_bm = score_part.measure_number_map(score_part.note_array()["onset_div"])
    except TypeError: # when the musicxml contains string measure numbers (i.e. special measures, X1, X2)
        measures_via_bm = None
        
    # Find measure numbers from musicxml file via note ids (without the repetition flags)
    measure_notes_dict = create_measures_notes_dict(score_path, piece)
    def remove_repetition_flag(note_id):
        return note_id[:-2]
    note_ids = score_part_df['id'].apply(remove_repetition_flag)

    measures_via_mxml = np.array([key for note_id in note_ids for key,
                item in measure_notes_dict.items() if note_id in item])
    # Cast musicxml measure numbers (which come as string) into numbers
    try:
        measures_via_mxml = measures_via_mxml.astype(np.int64)
    except ValueError:
        pass
    
    # Check if both arrays are equal
    if np.array_equal(measures_via_bm, measures_via_mxml):
        score_part_df['mn'] = measures_via_mxml
    else:
        score_part_df['mn_bm'] = measures_via_bm
        score_part_df['mn_mxml'] = measures_via_mxml
    
    return score_part_df

def process_unfolded_spart_with_mn_from_score(sonata_mv, score_path, match_path, save_path, create_max_unfolding=False, is_correction=False):
    """Create spart from a score, unfold it according to a given performance (or create the max unfolding if the piece has variations), add measure numbers and save it to save_path

    Parameters
    ----------
        sonata_mv : str
            name/number of the piece for which to create unfolded spart with mn
        create_max_unfolding : bool, optional
            whether to create max unfolding
        score_path : path
            path to score files
        match_path : path 
            path to matchfiles
        save_path : path
            where to save the created score part df
        is_correction : bool, optional
            flag used during dataset creation, to create new score part df in case the musicxml was manually corrected
    """
    sonata_no = sonata_mv[:3]
    spart_save_path = os.path.join(save_path,  f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_musicxml')
    if not os.path.exists(spart_save_path):
        os.makedirs(spart_save_path)
    
    if os.path.isfile(os.path.join(spart_save_path, 'spart_mn.csv')) and not is_correction:
        pass
    else:
        # For variations pieces we know that the performance corresponds to the maximum score unfolding
        if create_max_unfolding:
            spart_with_mn = create_unfolded_score_spart_with_mn(score_path, match_path, sonata_mv, unfold_structure='max')    
        else:
            spart_with_mn = create_unfolded_score_spart_with_mn(score_path, match_path, sonata_mv)
    
        # Save the spart with measure numbers as csv files
        if is_correction:
            pd.DataFrame(spart_with_mn).to_csv(os.path.join(spart_save_path, 'spart_mn_corr.csv'), index=False)
        else:
            pd.DataFrame(spart_with_mn).to_csv(os.path.join(spart_save_path, 'spart_mn.csv'), index=False)
    
    return None

# Preprocessing from match files 
def correct_onsets_in_mspart(sonata_mv, save_path, is_correction=False):
    """Sparts created from match do not always reflect pickup measures correctly, so this functions corrects those sparts
    
    Parameters
    ----------
        sonata_mv : str
            name/number of the piece for which to create unfolded spart with mn
        save_path : path
            where to save the created score part df
        is_correction : bool, optional
            flag used during dataset creation, to create new score part df in case the matchfile was manually corrected
    """
    
    sonata_no = sonata_mv[:3]
    spart_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_musicxml')
    mspart_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_match')
    
    if os.path.isfile(os.path.join(mspart_path, 'mspart_pickup_onsets.csv')) and not is_correction:
        pass
    else:
        spart = pd.read_csv(os.path.join(spart_path, 'spart_mn.csv'))
        if is_correction:
            mspart = pd.read_csv(os.path.join(mspart_path, 'mspart_corr.csv'))
        else: mspart = pd.read_csv(os.path.join(mspart_path, 'mspart.csv'))
        
        if spart['onset_beat'][0] < 0:
            pickup_beat_spart = spart['onset_beat'][0]
            pickup_quarter_spart = spart['onset_quarter'][0]
            pickup_beat_mspart = mspart['onset_beat'][0]
            pickup_quarter_mspart = mspart['onset_quarter'][0]
            
            if pickup_beat_spart != pickup_beat_mspart or pickup_quarter_spart != pickup_quarter_mspart:
                
                beat_diff = abs(pickup_beat_spart - pickup_beat_mspart)
                quarter_diff = abs(pickup_quarter_spart - pickup_quarter_mspart)
                mspart['onset_beat'] = mspart['onset_beat'] - beat_diff
                mspart['onset_quarter'] = mspart['onset_quarter'] - quarter_diff
                
                if not is_correction:
                    mspart.to_csv(os.path.join(mspart_path, 'mspart_pickup_onsets.csv'), index=False) 
                else:
                    mspart.to_csv(os.path.join(mspart_path, 'mspart_corr_pickup_onsets.csv'), index=False) 
    
    return None

def add_mn_to_mspart(sonata_mv, save_path, is_correction=False):
    """Add measure numbers to a score part from match by comparing each note onset in the match note array to the note onsets in score note array and taking the measure number of the note with the closest onset
    
    Parameters
    ----------
        sonata_mv : str
            name/number of the piece for which to create unfolded spart with mn
        save_path : path
            where to save the created score part df
        is_correction : bool, optional
            flag used during dataset creation, to create new score part df in case the matchfile was manually corrected
    """
    
    sonata_no = sonata_mv[:3]
    spart_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_musicxml')
    mspart_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_match')
    
    if os.path.isfile(os.path.join(mspart_path, 'mspart_mn.csv')) and not is_correction:
        pass
    
    else:
        spart = pd.read_csv(os.path.join(spart_path, 'spart_mn.csv'))
        if is_correction:
            if os.path.isfile(os.path.join(mspart_path, 'mspart_corr_pickup_onsets.csv')):
                mspart = pd.read_csv(os.path.join(mspart_path, 'mspart_corr_pickup_onsets.csv'))
            else:
                mspart = pd.read_csv(os.path.join(mspart_path, 'mspart_corr.csv'))
        else:
            if os.path.isfile(os.path.join(mspart_path, 'mspart_pickup_onsets.csv')):
                mspart = pd.read_csv(os.path.join(mspart_path, 'mspart_pickup_onsets.csv'))
            else:
                mspart = pd.read_csv(os.path.join(mspart_path, 'mspart.csv'))
            
    
        # # Find the closest onset_beat, check by matching duration_beat and pitch and add measure numbers to the match_spart
        tmp_merge = pd.merge_asof(mspart, spart, on=['onset_beat'], by=['pitch'], direction='nearest')
        
        if 'mn_bm' in tmp_merge.columns and 'mn_mxml' in tmp_merge.columns:
            if  not tmp_merge['mn_bm'].isnull().values.any():
                mspart['mn_bm'] = tmp_merge['mn_bm'].astype('int')
            else:
                mspart['mn_bm'] = tmp_merge['mn_bm']
            mspart['mn_mxml'] = tmp_merge['mn_mxml']
        else:
            mspart['mn'] = tmp_merge['mn'].astype('int')
        
        if not is_correction:
            # Save the match_spart to csv
            mspart.to_csv(os.path.join(mspart_path, 'mspart_mn.csv'), index=False)
        else:
            # Save the match_spart to csv
            mspart.to_csv(os.path.join(mspart_path, 'mspart_mn_corr.csv'), index=False)
            
    return None

def process_spart_with_mn_from_match(sonata_mv, match_path, save_path, is_correction=False):
    """Add measure numbers to a score part from match by comparing each note onset in the match note array to the note onsets in score note array and taking the measure number of the note with the closest onset
    
    Parameters
    ----------
        sonata_mv : str
            name/number of the piece for which to create unfolded spart with mn
        match_path : path
            path to matchfiles
        save_path : path
            where to save the created score part df
        is_correction : bool, optional
            flag used during dataset creation, to create new score part df in case the matchfile was manually corrected
    """
    sonata_no = sonata_mv[:3]
    mspart_save_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_match')
    if not os.path.exists(mspart_save_path):
        os.makedirs(mspart_save_path)
    
    if os.path.isfile(os.path.join(mspart_save_path, 'mspart_mn.csv')) and not is_correction:
        pass
        
    else:
        # Create a matchspart
        match_file = os.path.join(match_path, 'kv%s.match' %sonata_mv)
        _, _, match_part = pt.io.importmatch.load_match(match_file, create_part=True)
        match_spart = match_part[0].note_array()    
        # First time preprocessing
        if not is_correction:
            # Save the plain version from match
            pd.DataFrame(match_spart).to_csv(os.path.join(mspart_save_path, 'mspart.csv'), index=False)
        else:
            pd.DataFrame(match_spart).to_csv(os.path.join(mspart_save_path, 'mspart_corr.csv'), index=False)
        
        # Check if onsets need to be corrected
        correct_onsets_in_mspart(sonata_mv, save_path, is_correction=is_correction)
        # Add measure numbers
        add_mn_to_mspart(sonata_mv, save_path, is_correction=is_correction)
            
    return None
    
# Get both preprocessed sparts
def get_preprocessed_sparts(sonata_mv, save_path):
    
    sonata_no = sonata_mv[:3]
    spart_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_musicxml')
    mspart_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_from_match')

    spart_corr, mspart_corr = False, False
    if os.path.isfile(os.path.join(spart_path, 'spart_mn_corr.csv')):
        spart = pd.read_csv(os.path.join(spart_path, 'spart_mn_corr.csv'))
        spart_corr = True
    else:
        spart = pd.read_csv(os.path.join(spart_path, 'spart_mn.csv'))
    
    if os.path.isfile(os.path.join(mspart_path, 'mspart_mn_corr.csv')):
        mspart = pd.read_csv(os.path.join(mspart_path, 'mspart_mn_corr.csv'))
        mspart_corr = True
    else:
        mspart = pd.read_csv(os.path.join(mspart_path, 'mspart_mn.csv'))
    
    sparts_save_path = os.path.join(save_path, f'KV{sonata_no}', f'kv{sonata_mv}', 'sparts_preprocessed')
    # if it's a file correction we want to override
    if not os.path.isfile(os.path.join(sparts_save_path, 'spart.csv')) or spart_corr == True:
        spart.to_csv(os.path.join(sparts_save_path, 'spart.csv'), index=None)
    if not os.path.isfile(os.path.join(sparts_save_path, 'mspart.csv')) or mspart_corr == True:
        mspart.to_csv(os.path.join(sparts_save_path, 'mspart.csv'), index=None)

    return spart, mspart