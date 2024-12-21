import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import scipy

from MAGNN_utils.preprocess import (
    assign_array_index,
    export_index2dat,
    generate_long_relationship_array,
    generate_triplet_array,
    lexicographical_sort,
    load_and_concat_files,
    map_indices_to_dataframe,
    sample_edges,
    save_split_data2npz,
    split_date,
)

# list all file paths
file_path = os.path.join(os.getcwd(), "../data", "MAGNN_data")
# microbe-disease files
disbiome_microd_path = os.path.join(file_path, "disbiome_taxid_mondo.dat")
gmmad2_microd_path = os.path.join(file_path, "gmmad2_taxid_mondo.dat")
# microbe-metabolite files
gmmad2_micrometa_path = os.path.join(file_path, "gmmad2_taxid_met.dat")
hmdb_micrometa_path = os.path.join(file_path, "hmdb_taxid_met.dat")
# metabolite-disease file
hmdb_metad_path = os.path.join(file_path, "hmdb_met_disease.dat")

# load each dataset and sample the datasets
microd_df = load_and_concat_files(
    [disbiome_microd_path, gmmad2_microd_path], column_names=["Microbe", "Disease"]
)
microd_frac = sample_edges(dataset=microd_df, fraction=0.1)
micrometa_df = load_and_concat_files(
    [gmmad2_micrometa_path, hmdb_micrometa_path], column_names=["Microbe", "Metabolite"]
)
micrometa_frac = sample_edges(dataset=micrometa_df, fraction=0.1)
metad_df = load_and_concat_files([hmdb_metad_path], column_names=["Metabolite", "Disease"])
metad_frac = sample_edges(dataset=metad_df, fraction=1.0)

# assign index to each node
microbes1 = microd_frac["Microbe"].unique()
microbes2 = micrometa_frac["Microbe"].unique()
all_microbes = assign_array_index(
    [microbes1, microbes2], col_name="Microbe", index_name="MicrobeIdx"
)
export_index2dat(all_microbes, "data/sampled/microbe_index.dat")
d1 = microd_frac["Disease"].unique()
d2 = metad_frac["Disease"].unique()
all_diseases = assign_array_index([d1, d2], col_name="Disease", index_name="DiseaseIdx")
export_index2dat(all_diseases, "data/sampled/disease_index.dat")
metabolites1 = micrometa_frac["Metabolite"].unique()
metabolites2 = metad_frac["Metabolite"].unique()
all_metabolites = assign_array_index(
    [metabolites1, metabolites2], col_name="Metabolite", index_name="MetaboliteIdx"
)
export_index2dat(all_metabolites, "data/sampled/metabolite_index.dat")

microd = map_indices_to_dataframe(
    input_df=microd_frac,
    col1="Microbe",
    col2="Disease",
    index_df1=all_microbes,
    index_col1="Microbe",
    index_col1_idx="MicrobeIdx",
    index_df2=all_diseases,
    index_col2="Disease",
    index_col2_idx="DiseaseIdx",
)
export_index2dat(microd, "data/sampled/microbe_disease.dat")
micrometa = map_indices_to_dataframe(
    input_df=micrometa_frac,
    col1="Microbe",
    col2="Metabolite",
    index_df1=all_microbes,
    index_col1="Microbe",
    index_col1_idx="MicrobeIdx",
    index_df2=all_metabolites,
    index_col2="Metabolite",
    index_col2_idx="MetaboliteIdx",
)
export_index2dat(micrometa, "data/sampled/microbe_metabolite.dat")
metad = map_indices_to_dataframe(
    input_df=metad_frac,
    col1="Metabolite",
    col2="Disease",
    index_df1=all_metabolites,
    index_col1="Metabolite",
    index_col1_idx="MetaboliteIdx",
    index_df2=all_diseases,
    index_col2="Disease",
    index_col2_idx="DiseaseIdx",
)
export_index2dat(metad, "data/sampled/metabolite_disease.dat")

microd = pd.read_csv(
    "data/sampled/microbe_disease.dat",
    sep="\t",
    encoding="utf-8",
    header=None,
    names=["MicrobeIdx", "DiseaseIdx"],
)
micrometa = pd.read_csv(
    "data/sampled/microbe_metabolite.dat",
    sep="\t",
    encoding="utf-8",
    header=None,
    names=["MicrobeIdx", "MetaboliteIdx"],
)
metad = pd.read_csv(
    "data/sampled/metabolite_disease.dat",
    sep="\t",
    encoding="utf-8",
    header=None,
    names=["MetaboliteIdx", "DiseaseIdx"],
)

md_train, md_val, md_test = split_date(microd, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
save_split_data2npz(md_train, md_val, md_test, "data/sampled/micro_disease_train_val_test_idx.npz")

# training: 70%, validation: 20%, testing: 10%
train_val_test_idx = np.load("data/sampled/micro_disease_train_val_test_idx.npz")
train_idx = train_val_test_idx["train"]
val_idx = train_val_test_idx["val"]
test_idx = train_val_test_idx["test"]

# reset microbe-disease index
microbe_disease = microd.loc[train_idx].reset_index(drop=True)
microbe_disease.head()
print(f"Length of Training data: {len(microbe_disease)}")

save_prefix = "data/sampled/preprocessed/"

num_microbe = 7180
num_disease = 771
num_metabolite = 23665

# build adjacency matrix
# 0 for microbe, 1 for disease, 2 for metabolite
dim = num_microbe + num_disease + num_metabolite

type_mask = np.zeros(dim, dtype=np.int16)
type_mask[num_microbe : num_microbe + num_disease] = 1
type_mask[num_microbe + num_disease :] = 2

adjM = np.zeros((dim, dim), dtype=np.int16)
for _, row in microd.iterrows():
    microID = row["MicrobeIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    adjM[microID, diseaseID] = 1
    adjM[diseaseID, microID] = 1
for _, row in micrometa.iterrows():
    microID = row["MicrobeIdx"]
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    adjM[microID, metID] = 1
    adjM[metID, microID] = 1
for _, row in metad.iterrows():
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    adjM[metID, diseaseID] = 1
    adjM[diseaseID, metID] = 1

# map each microbe to a list of diseases within adjM and remove empty arrays
# adjM[microbe, diseases]
microbe_disease_list = {
    i: adjM[i, num_microbe : num_microbe + num_disease].nonzero()[0].astype(np.int16)
    for i in range(num_microbe)
}
microbe_disease_list = {i: v for i, v in microbe_disease_list.items() if v.size > 0}

# map each disease to a list of microbes within adjM and remove empty arrays
# adjM[disease, microbes]
disease_microbe_list = {
    i: adjM[num_microbe + i, :num_microbe].nonzero()[0].astype(np.int16) for i in range(num_disease)
}
disease_microbe_list = {i: v for i, v in disease_microbe_list.items() if v.size > 0}

# map each metabolite to a list of diseases within adjM and remove empty arrays
# adjM[metabolite, diseases]
metabolite_disease_list = {
    i: adjM[num_microbe + num_disease + i, num_microbe : num_microbe + num_disease]
    .nonzero()[0]
    .astype(np.int16)
    for i in range(num_metabolite)
}
metabolite_disease_list = {i: v for i, v in metabolite_disease_list.items() if v.size > 0}

# map each disease to a list of metabolites within adjM and remove empty arrays
# adjM[disease, metabolites]
disease_metabolite_list = {
    i: adjM[num_microbe + i, num_microbe + num_disease :].nonzero()[0].astype(np.int16)
    for i in range(num_disease)
}
disease_metabolite_list = {i: v for i, v in disease_metabolite_list.items() if v.size > 0}

# map each microbe to a list of metabolites within adjM and remove empty arrays
# adjM[microbe, metabolites]
microbe_metabolite_list = {
    i: adjM[i, num_microbe + num_disease :].nonzero()[0].astype(np.int16)
    for i in range(num_microbe)
}
microbe_metabolite_list = {i: v for i, v in microbe_metabolite_list.items() if v.size > 0}

# map each metabolite to a list of microbes within adjM and remove empty arrays
# adjM[metabolite, microbes]
metabolite_microbe_list = {
    i: adjM[num_microbe + num_disease + i, :num_microbe].nonzero()[0].astype(np.int16)
    for i in range(num_metabolite)
}
metabolite_microbe_list = {i: v for i, v in metabolite_microbe_list.items() if v.size > 0}

# 0-1-0 (microbe-disease-microbe)
# remove the same metapath types with reverse order. e.g., (1, 0, 2) and (2, 0, 1) are the same
# remove path includes the same microbe1 and microbe2 (same 1st and last element). e.g., (1, 4, 1) and (0, 4, 0) are removed
microbe_disease_microbe = generate_triplet_array(disease_microbe_list, dtype=np.int16)
microbe_disease_microbe[:, 1] += num_microbe
microbe_disease_microbe = lexicographical_sort(microbe_disease_microbe, [0, 2, 1])

# 0-2-0 (microbe-metabolite-microbe)
microbe_metabolite_microbe = generate_triplet_array(metabolite_microbe_list, dtype=np.int16)
microbe_metabolite_microbe[:, 1] += num_microbe + num_disease
microbe_metabolite_microbe = lexicographical_sort(microbe_metabolite_microbe, [0, 2, 1])

# 1-0-1 (disease-microbe-disease)
disease_microbe_disease = generate_triplet_array(microbe_disease_list, dtype=np.int16)
disease_microbe_disease[:, (0, 2)] += num_microbe
disease_microbe_disease = lexicographical_sort(disease_microbe_disease, [0, 2, 1])

# 1-2-1 (disease-metabolite-disease)
disease_metabolite_disease = generate_triplet_array(metabolite_disease_list, dtype=np.int16)
disease_metabolite_disease[:, (0, 2)] += num_microbe
disease_metabolite_disease[:, 1] += num_microbe + num_disease
disease_metabolite_disease = lexicographical_sort(disease_metabolite_disease, [0, 2, 1])

# 2-0-2 (metabolite-microbe-metabolite)
metabolite_microbe_metabolite = generate_triplet_array(microbe_metabolite_list, dtype=np.int16)
metabolite_microbe_metabolite[:, (0, 2)] += num_microbe + num_disease
metabolite_microbe_metabolite = lexicographical_sort(metabolite_microbe_metabolite, [0, 2, 1])

# 2-1-2 (metabolite-disease-metabolite)
metabolite_disease_metabolite = generate_triplet_array(disease_metabolite_list, dtype=np.int16)
metabolite_disease_metabolite[:, (0, 2)] += num_microbe + num_disease
metabolite_disease_metabolite[:, 1] += num_microbe
metabolite_disease_metabolite = lexicographical_sort(metabolite_disease_metabolite, [0, 2, 1])

# 1-0-2-0-1 (disease-microbe-metabolite-microbe-disease)
d_micro_meta_micro_d = generate_long_relationship_array(
    relational_list=microbe_disease_list,
    intermediate_triplet=microbe_metabolite_microbe,
    num_offset2=num_microbe,
    scaling_factor=1.0,
)

d_micro_meta_micro_d = lexicographical_sort(d_micro_meta_micro_d, [0, 2, 1, 3, 4])

# 0-1-2-1-0 (microbe-disease-metabolite-disease-microbe)
micro_d_meta_d_micro = generate_long_relationship_array(
    relational_list=disease_microbe_list,
    intermediate_triplet=disease_metabolite_disease,
    num_offset1=num_microbe,
    scaling_factor=0.3,
)

micro_d_meta_d_micro = lexicographical_sort(micro_d_meta_d_micro, [0, 2, 1, 3, 4])

# 0-2-1-2-0 (microbe-metabolite-disease-metabolite-microbe)
micro_meta_d_meta_micro = generate_long_relationship_array(
    relational_list=metabolite_microbe_list,
    intermediate_triplet=metabolite_disease_metabolite,
    num_offset1=(num_microbe + num_disease),
)

micro_meta_d_meta_micro = lexicographical_sort(micro_meta_d_meta_micro, [0, 2, 1, 3, 4])

# 1-2-0-2-1 (disease-metabolite-microbe-metabolite-disease)
d_meta_micro_meta_d = generate_long_relationship_array(
    relational_list=metabolite_disease_list,
    intermediate_triplet=metabolite_microbe_metabolite,
    num_offset1=(num_microbe + num_disease),
    num_offset2=num_microbe,
    scaling_factor=0.5,
)

d_meta_micro_meta_d = lexicographical_sort(d_meta_micro_meta_d, [0, 2, 1, 3, 4])

expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
    [(1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1)],
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + "{}".format(i)).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {
    (0, 1, 0): microbe_disease_microbe,
    (0, 1, 2, 1, 0): micro_d_meta_d_micro,
    (0, 2, 0): microbe_metabolite_microbe,
    (0, 2, 1, 2, 0): micro_meta_d_meta_micro,
    (1, 0, 1): disease_microbe_disease,
    (1, 0, 2, 0, 1): d_micro_meta_micro_d,
    (1, 2, 0, 2, 1): d_meta_micro_meta_d,
    (1, 2, 1): disease_metabolite_disease,
}

# write all things
target_idx_lists = [np.arange(num_microbe), np.arange(num_disease)]
offset_list = [0, num_microbe]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(
            save_prefix + "{}/".format(i) + "-".join(map(str, metapath)) + "_idx.pickle", "wb"
        ) as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while (
                    right < len(edge_metapath_idx_array)
                    and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]
                ):
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        # np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        with open(
            save_prefix + "{}/".format(i) + "-".join(map(str, metapath)) + ".adjlist", "w"
        ) as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while (
                    right < len(edge_metapath_idx_array)
                    and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]
                ):
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write("{} ".format(target_idx) + " ".join(neighbors) + "\n")
                else:
                    out_file.write("{}\n".format(target_idx))
                left = right

scipy.sparse.save_npz(save_prefix + "adjM.npz", scipy.sparse.csr_matrix(adjM))
np.save(save_prefix + "node_types.npy", type_mask)

# output microbe_disease.npy
microbe_disease = pd.read_csv(
    "data/sampled/microbe_disease.dat",
    encoding="utf-8",
    delimiter="\t",
    names=["MicrobeID", "DiseaseID"],
)
microbe_disease = microbe_disease[["MicrobeID", "DiseaseID"]].to_numpy()
np.save(save_prefix + "microbe_disease.npy", microbe_disease)

# output positive and negative samples for training, validation and testing

np.random.seed(453289)
save_prefix = "data/sampled/preprocessed/"
num_microbe = 7180
num_disease = 771
microbe_disease = np.load("data/sampled/preprocessed/microbe_disease.npy")
train_val_test_idx = np.load("data/sampled/micro_disease_train_val_test_idx.npz")
train_idx = train_val_test_idx["train"]
val_idx = train_val_test_idx["val"]
test_idx = train_val_test_idx["test"]

neg_candidates = []
counter = 0
for i in range(num_microbe):
    for j in range(num_disease):
        if counter < len(microbe_disease):
            if i == microbe_disease[counter, 0] and j == microbe_disease[counter, 1]:
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)

idx = np.random.choice(len(neg_candidates), len(val_idx) + len(test_idx), replace=False)
val_neg_candidates = neg_candidates[sorted(idx[: len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx) :])]

train_microbe_disease = microbe_disease[train_idx]
train_neg_candidates = []
counter = 0
for i in range(num_microbe):
    for j in range(num_disease):
        if counter < len(train_microbe_disease):
            if i == train_microbe_disease[counter, 0] and j == train_microbe_disease[counter, 1]:
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

np.savez(
    save_prefix + "train_val_test_neg_user_artist.npz",
    train_neg_micro_dis=train_neg_candidates,
    val_neg_micro_dis=val_neg_candidates,
    test_neg_micro_dis=test_neg_candidates,
)
np.savez(
    save_prefix + "train_val_test_pos_user_artist.npz",
    train_pos_micro_dis=microbe_disease[train_idx],
    val_pos_micro_dis=microbe_disease[val_idx],
    test_pos_micro_dis=microbe_disease[test_idx],
)
