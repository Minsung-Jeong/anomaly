import Levenshtein
import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
from Bio.SubsMat import MatrixInfo
from scipy import stats
import os


cap_sub_score_zero = True
ddg_filna_score = -0.25
submit_col = 'rank_pow'
sigmoid_norm_factor = 3

class paths:
    TRAIN = "./kaggle/enzyme/train.csv"
    TEST = "./kaggle/enzyme/test.csv"
    SUBMISSION = "./kaggle/enzyme/sample_submission.csv"
    PDB_FILE = "./kaggle/enzyme/wildtype_structure_prediction_af2.pdb"

base = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'

# add mutation information to the test data
# source: https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/354783
def get_mutation_info(_row, _wildtype=base):
    terminology_map = {"replace":"substitution", "insert":"insertion", "delete":"deletion"}
    req_edits = Levenshtein.editops(_wildtype, _row["protein_sequence"])
    _row["n_edits"] = len(req_edits)

    if _row["n_edits"]==0:
        _row["edit_type"] = _row["edit_idx"] = _row["old_aa"] = _row["new_aa"] = pd.NA
    else:
        _row["edit_type"] = terminology_map[req_edits[0][0]]
        _row["edit_idx"] = req_edits[0][1]
        _row["old_aa"] = _wildtype[_row["edit_idx"]]
        _row["new_aa"] = _row["protein_sequence"][req_edits[0][2]] if _row["edit_type"]!="deletion" else pd.NA
    return _row

def revert_to_wildtype(protein_sequence, edit_type, edit_idx, old_aa, new_aa):
    if pd.isna(edit_type):
        return protein_sequence
    elif edit_type!="insertion":
        new_wildtype_base = protein_sequence[:edit_idx]
        if edit_type=="deletion":
            new_wildtype=new_wildtype_base+old_aa+protein_sequence[edit_idx:]
        else:
            new_wildtype=new_wildtype_base+old_aa+protein_sequence[edit_idx+1:]
    else:
        new_wildtype=protein_sequence[:edit_idx]+old_aa+protein_sequence[edit_idx:]
    return new_wildtype


#helper function
def read_list_from_file(list_file):
    with open(list_file) as f:
        lines  = f.readlines()
    return lines


# modify test data
os.chdir('C://data_minsung')
test_df = pd.read_csv(paths.TEST)
test_df = test_df.apply(get_mutation_info, axis=1)
test_df.loc[test_df.edit_type.isna(), 'edit_type'] = 'nothing'
test_df.head()

# pdb data
pdb_df =  PandasPdb().read_pdb(paths.PDB_FILE)
pdb_df.df.keys()

atom_df = pdb_df.df['ATOM']
atom_df['residue_number_0based'] = atom_df['residue_number'] - 1
map_number_to_b = atom_df.groupby('residue_number_0based').b_factor.mean()
test_df['b_factor'] = test_df.edit_idx.map(map_number_to_b).fillna(0)
test_df.loc[test_df['edit_type']=='deletion', 'new_aa'] = '-'
test_df.loc[test_df['edit_type']=='insertion', 'new_aa'] = '+'
test_df.loc[:,'mut_string'] =  test_df.old_aa+test_df.edit_idx.astype(str)+test_df.new_aa
test_df.head()

# get ddg
os.getcwd()
ddg = read_list_from_file('./kaggle/enzyme/extra/wildtype_structure_prediction_af2.deepddg.ddg.txt')
header = ddg[0]
data = [s.split() for s in ddg[1:]]

df = pd.DataFrame(data, columns = ['chain', 'WT', 'ResID', 'Mut', 'ddG'])
df.ddG = df.ddG.astype(np.float32)
df.ResID = df.ResID.astype(int)
df.loc[:,'location'] = df.ResID -1  #change to 0-indexing
df.loc[:,'mut_string'] = df.WT+df.location.astype(str)+df.Mut
df.head()

# merge data to test_df
test_df = test_df.merge(df[['ddG','mut_string']], on='mut_string', how='left')
test_df.loc[test_df['ddG'].isna(), 'ddG'] = ddg_filna_score
test_df.head()

# add sub_score
sub_scores = []
sub_mat = MatrixInfo.blosum100
for i in range(len(test_df)):
    mut_type = test_df.edit_type.values[i]
    if mut_type == 'substitution':
        try:
            sub_score = sub_mat[(test_df.old_aa.values[i], test_df.new_aa.values[i])]
        except KeyError:
            sub_score = sub_mat[(test_df.new_aa.values[i], test_df.old_aa.values[i])]
    elif mut_type == 'nothing':
        sub_score = 0
    else:
        sub_score = -10
    sub_scores.append(sub_score)

# add sub_score, ddg_rank, b_factor_rank, b_factor_adj_rank, sub_score_rank to test_df
test_df['sub_score'] = sub_scores
if cap_sub_score_zero:
    test_df.loc[test_df['sub_score'] > 0, 'sub_score'] = 0
test_df['score_adj'] = [ 1 - (1 / (1+np.exp(-x/sigmoid_norm_factor))) for x in sub_scores]
test_df['b_factor_adj'] = test_df['b_factor'] * test_df['score_adj']
test_df.head(5)

test_df['ddG_rank'] = stats.rankdata(test_df['ddG'])
test_df['b_factor_rank'] = stats.rankdata(-test_df['b_factor'])
test_df['b_factor_adj_rank'] = stats.rankdata(-test_df['b_factor_adj'])
test_df['sub_score_rank'] = stats.rankdata(test_df['sub_score'])
test_df.head()

test_df['rank_pow'] = test_df.apply(lambda x: np.power(x['b_factor_rank'] * x['sub_score_rank'] * x['ddG_rank'], 1/3), axis=1)
# test_df['rank_pow'] = test_df.apply(lambda x: np.power(x['b_factor_adj_rank'] * x['sub_score_rank'] * x['ddG_rank'], 1/3), axis=1)
#test_df['rank_pow'] = test_df.apply(lambda x: np.power(x['b_factor_rank'] * x['b_factor_adj_rank'], 1/2), axis=1)
test_df.head()

test_df.sort_values('rank_pow')

assert not test_df[submit_col].isna().any()
submit_df = pd.DataFrame({
    'seq_id': test_df.seq_id.values,
    'tm': test_df[submit_col].values,
})
submit_df.tm = submit_df.tm.fillna(0)
submit_df.to_csv('deepddg-ddg.csv', index=False)
submit_df.head()