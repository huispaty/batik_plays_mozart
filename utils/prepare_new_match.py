import os
import pandas as pd
import shutil
import csv
import numpy as np
import partitura as pt
import random

def make_perf2score_alignment_dirs(repo_dir, dir_name, pieces):
    perf2score_align_dir = os.path.join(repo_dir, dir_name)
    if not os.path.exists(perf2score_align_dir):
        os.mkdir(perf2score_align_dir)
    
    for p in pieces:
        # Create the necessary dirs
        sonata_dir = os.path.join(perf2score_align_dir, f'KV{p[:3]}')
        if not os.path.exists(sonata_dir):
            os.mkdir(os.path.join(sonata_dir))
            movements = [f'kv{p[:3]}_{mv}' for mv in range(1, 4)]

            # prealign_working_dirs = ['sparts_from_match', 'sparts_from_musicxml', 'sparts_preprocessed', 'aligned_note_ids']
            for mv in movements:
                os.mkdir(os.path.join(sonata_dir, mv))
                # for wdir in prealign_working_dirs:
                #     os.mkdir(os.path.join(sonata_dir, mv, wdir))
        else:
            pass
    
    test_piece = random.choice(pieces)
    test_piece_path = os.path.join(perf2score_align_dir, f'KV{test_piece[:3]}', f'kv{test_piece}')
    assert os.path.exists(test_piece_path)

def update_perf2score_alignment(piece, new_perf2score_alignment_dir, old_perf2score_alignment, score2score_alignment): 
    """Create performance-score alignments by replacing the score part in the old (perf2score) alignment with the updated score part using the score2score alignments.
    
    Parameters
    ----------
        piece : str
            current piece in the form <SONATA_NUM>_<movement>
        new_perf2score_alignment_dir : path
            path to the updated perf2score alignments
        old_perf2score_alignment : list
            list of perf_note-score_note alignments of type match, insertion or deletion
        score2score_alignment : list
            list of score_note-score_note alignments
        
    Returns:
        alignment : list
            updated perf2score alignment, i.e. list of dictionaries where each dict reflects an perf2score note alignment (of type match, insertion or deletion)
    """

    # Save alignment
    with open(os.path.join(new_perf2score_alignment_dir, 'alignment.csv'), 'w+') as af:
        writer = csv.writer(af)
        writer.writerow(['label', 'score_id', 'performance_id'])
        for al in old_perf2score_alignment:
            if al['label'] == 'match':
                writer.writerow(al.values())
            else:
                vals = list(al.values())
                if al['label'] == 'insertion':
                    vals.insert(1,'undefined')
                elif al['label'] == 'deletion':
                    vals.append('undefined')
                writer.writerow(vals)
    # Convert the old alignment csv into a df
    old_perf2score_alignment = pd.read_csv(os.path.join(new_perf2score_alignment_dir, 'alignment.csv'))
    
    # Create the updated perf2score alignment
    # Check how match score notes and xml score notes are aligned
    msnotes_not_aligned = score2score_alignment['id_m'][score2score_alignment['id_s'] == 'undefined']
    snotes_not_aligned = score2score_alignment['id_s'][score2score_alignment['id_m'] == 'undefined']
    msnotes_mapped_to_undefined, snotes_mapped_to_undefined = False, False
    # How to resolve score2score (match_score, xml_score) -> perf2score (perf, xml_score)
    # Case 1: all match score notes and xml score notes are aligned (no unaligned values) -> overwrite match score ids 1:1 with xml score ids
    # Case 2: all match score notes are aligned but 1+ xml score note not aligned (undefined,snote-id) -> add new deletion for unaligned xml score note
    if msnotes_not_aligned.shape[0] == 0 and snotes_not_aligned.shape[0] != 0:
        snotes_mapped_to_undefined = True
    # Case 3: all xml score notes are aligned but 1+ match score note not aligned -> two possibilities
    # 1) Old perf2score alignment for match score note was deletion,msnote-id, -> entry will be deleted completely
    # 2) Old perf2score alignment was match,msnote-id,psnote-id -> convert alignment type match into type insertion, omit msnote-id
    elif snotes_not_aligned.shape[0] == 0 and msnotes_not_aligned.shape[0]!= 0:
        msnotes_mapped_to_undefined = True
    # Case 4: some match score notes and some xml score notes are not aligned
    # For snotes not aligned: add new deletions 
    # For msnotes not aligned: will disappear completely (if originally msnote belonged to a deletion), otherwise will be converted to insertion (similar to case 3)
    else:
        snotes_mapped_to_undefined, msnotes_mapped_to_undefined = True, True
        
    # Create the updated perf2score alignment
    # Case 1: all old match score notes and xml score notes are aligned
    # &
    # Case 3: all snotes are aligned but 1+ msnote not aligned -> two possibilities
    # 1) Original alignment was deletion,msnote-id, -> entry will be deleted completely
    # 2) Original alignment was match,msnote-id,psnote-id -> match will be converted to insertion
    if (not snotes_mapped_to_undefined and not msnotes_mapped_to_undefined) or (not snotes_mapped_to_undefined and msnotes_mapped_to_undefined):
        new_perf2score_alignment = old_perf2score_alignment.merge(score2score_alignment, left_on='score_id', right_on='id_m', how='left')
        # Replace match score id with matched mxml score id, drop non-used columns
        new_perf2score_alignment['score_id'] = new_perf2score_alignment['id_s'].str.replace('s','n')
        new_perf2score_alignment.drop(columns=['id_m', 'id_s'], inplace=True)
        assert old_perf2score_alignment.shape[0] == new_perf2score_alignment.shape[0]
    # Case 2: all msnotes are aligned but 1+ snote not aligned
    # &
    # Case 4: some msnotes and some snotes not aligned
    elif (snotes_mapped_to_undefined and not msnotes_mapped_to_undefined) or (snotes_mapped_to_undefined and msnotes_mapped_to_undefined):
        new_perf2score_alignment_part = old_perf2score_alignment.merge(score2score_alignment[score2score_alignment['id_m'] != 'undefined'], left_on='score_id', right_on='id_m', how='left')
        # -> save non-mapped snotes as additional deletions
        additional_deletions = score2score_alignment[score2score_alignment['id_m'] == 'undefined']
        # Replace match score id with matched mxml score id, drop non-used columns
        new_perf2score_alignment_part['score_id'] = new_perf2score_alignment_part['id_s'].str.replace('s','n')
        new_perf2score_alignment_part.drop(columns=['id_m', 'id_s'], inplace=True)
        # Change the additional deletions so they match the format of the alignment
        additional_deletions['label'] = 'deletion'
        additional_deletions.rename(columns={"id_m": "performance_id", "id_s": "score_id"}, inplace=True)
        additional_deletions['score_id'] = additional_deletions['score_id'].str.replace('s','n')
        new_perf2score_alignment = pd.concat([new_perf2score_alignment_part, additional_deletions[['label', 'score_id', 'performance_id']]])
        assert new_perf2score_alignment.shape[0] == old_perf2score_alignment.shape[0] + additional_deletions.shape[0]
        
    # Check if notes from different score versions have been aligned
    if new_perf2score_alignment['score_id'].str.contains('v').any():
        # print(f'Saved perf2score alignment for different score versions for {piece}')
        new_perf2score_alignment.to_csv(os.path.join(new_perf2score_alignment_dir, 'alignment_svdiff.csv'), index=None)
        new_perf2score_alignment['score_id'] = new_perf2score_alignment['score_id'].str.replace('v','')
            
    # Save updated perf2score alignment
    new_perf2score_alignment.to_csv(os.path.join(new_perf2score_alignment_dir, 'alignment.csv'), index=None)
    # print(f'Updated perf2score alignment for piece {piece}')
    
    new_perf2score_alignment = os.path.join(new_perf2score_alignment_dir, 'alignment.csv')
    reader = csv.DictReader(open(new_perf2score_alignment, 'r'))
    alignment = list()
    for dictionary in reader:
        # remove undefined-undefined alignments
        dict = {k: v for k, v in dictionary.items() if v}
        
        if dict['label'] == 'match':
            if dict['score_id'] == 'undefined' and dict['performance_id'] != 'undefined':
                # reflects Case 3b, when match becomes insertion
                dict['label'] = 'insertion'
                dict.pop('score_id')
        elif dict['label'] == 'deletion':
            # remove wrong deletions
            # some note deletions (deletion,snote-id) from the old matchfiles are not reflected in the matchfile spart -> they are not matched then, and will have performance_id 'undefined'
            if 'score_id' not in dict.keys() and dict['performance_id'] == 'undefined':
                continue
            else:
                dict.pop('performance_id')
        elif dict['label'] == 'insertion':
            pass
        alignment.append(dict)
    
    # save the alignment in pt style/format
    pd.DataFrame.from_dict(alignment).to_csv(os.path.join(new_perf2score_alignment_dir, 'alignment.csv'), index=False)
    
    return alignment

def create_matchfile(piece, score2score_alignment_dir, perf2score_alignment_dir, old_alignments_dir, musicXML_score_dir, midi_performance_dir, dataset='BatikPlaysMozart'):
    """ Create new matchiles for the <Batik-Plays-Mozart> Dataset.

    Arguments:
    ----------
        piece : str
            the piece to generate a matchfile for, in the format <SONATA_NUM>_<movement>
        score2score_alignment_dir : path
            path to score2score alignments
        perf2score_alignment_dir : path
            path to perf2score alignments
        old_alignments_dir : path
            path to all scores in musicXML format
        musicXML_score_dir : path
            path to all scores in musicXML format
        midi_performance_dir : path
            path to all performances in midi format
        dataset : str
            name of dataset for meta information in a matchfile (here: 'BatikPlaysMozart')

    """
    
    if dataset == 'BatikPlaysMozart':    
        performer = 'Roland Batik'
        composer = 'Wolfgang Amadeus Mozart'

    # Get score2score note alignments, unfolded score part from MusicXML and score musicxml file
    score2score_alignment_path = os.path.join(os.path.join(score2score_alignment_dir, f'KV{piece[:-2]}', f'kv{piece}', 'aligned_note_ids'))
    manually_corr_alignment = os.path.join(score2score_alignment_path, 'aligned_note_ids_manual_corr.csv')
    if os.path.isfile(manually_corr_alignment):
        aligned_snote_ids = pd.read_csv(manually_corr_alignment)
    else: aligned_snote_ids = pd.read_csv(os.path.join(score2score_alignment_path, 'aligned_note_ids.csv'))    
    # Get perf2score alignment from old matchfiles
    old_perf2score_alignment_file = os.path.join(old_alignments_dir, f'kv{piece}.match')
    match_perf, old_perf2score_alignment = pt.load_match(old_perf2score_alignment_file)   
    # Update perf2score alignment
    new_perf2score_alignment_dir = os.path.join(perf2score_alignment_dir, f'KV{piece[:3]}', f'kv{piece}')
    new_perf2score_alignment = update_perf2score_alignment(piece, new_perf2score_alignment_dir, old_perf2score_alignment, score2score_alignment=aligned_snote_ids)

    # Add missing pedaling information to old performances
    match_ppart = match_perf.performedparts[0]
    controls = pt.load_performance_midi(os.path.join(midi_performance_dir, f'kv{piece}.mid')).performedparts[0].controls
    match_ppart.controls = controls
    
    # Get score information: spart and score xml file
    score_musicxml = os.path.join(musicXML_score_dir, f'kv{piece}.musicxml')
    score = pt.load_musicxml(score_musicxml)
    score_part = score.parts[0]
    if piece in ['284_3', '331_1', '331_3']: # the variations pieces
        score_part = pt.score.unfold_part_maximal(score[0], update_ids=True)
    else: score_part = pt.score.unfold_part_alignment(score_part, new_perf2score_alignment)
    # Save score and spart
    shutil.copyfile(score_musicxml, os.path.join(new_perf2score_alignment_dir, f'kv{piece}.musicxml'))
    pd.DataFrame(score_part.note_array()).to_csv(os.path.join(new_perf2score_alignment_dir, 'spart.csv'), index=None)
    # Check if there are score-perf alignments capturing a special score different (original version, editors note etc.)
    diff_score_version_notes = None
    diff_score_versions = os.path.join(new_perf2score_alignment_dir, 'alignment_svdiff.csv')
    if os.path.isfile(diff_score_versions):
        diff_score_version_notes = pd.read_csv(diff_score_versions)
        diff_score_version_notes = diff_score_version_notes[diff_score_version_notes['score_id'].str.contains('v', na=False)]['score_id']
        diff_score_version_notes = diff_score_version_notes.str.replace('v', '').to_list()
        # print(f'piece {piece} has different score version alignments: {diff_score_version_notes}')
    
    # Create matchfile
    # define piece-wise information
    piece_info = f'Sonata KV{piece[:3]}, {piece[-1]}. Movement'
    score_fn = f'kv{piece[:3]}_{piece[-1]}.musicxml'
    performance_fn = f'kv{piece}.mid'
    # exporting to match file
    new_match_file = pt.io.exportmatch.matchfile_from_alignment(new_perf2score_alignment, match_ppart, score_part, performer=performer, composer=composer, piece=piece_info, score_filename=score_fn, performance_filename=performance_fn, assume_part_unfolded=True, tempo_indication=score.movement_title, diff_score_version_notes=diff_score_version_notes)
    # save in perf2score dir
    new_match_fn = os.path.join(new_perf2score_alignment_dir, f'kv{piece}.match')
    new_match_file.write(new_match_fn)
    
    # Save ppart and performance midi
    perf, _ = pt.load_match(new_match_fn)
    match_ppart = perf.performedparts[0]
    # Save perf and ppart
    pd.DataFrame(match_ppart.note_array()).to_csv(os.path.join(new_perf2score_alignment_dir, 'ppart.csv'), index=None)    
    pt.save_performance_midi(match_ppart, os.path.join(new_perf2score_alignment_dir, f'kv{piece}.mid'), merge_tracks_save=True)

    return None