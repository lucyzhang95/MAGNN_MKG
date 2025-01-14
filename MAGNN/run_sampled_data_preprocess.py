import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import scipy

from MAGNN_utils.preprocess import (
    generate_long_relationship_array,
    generate_triplet_array,
    lexicographical_sort,
    save_split_data2npz,
    split_date,
)

# load 3 relation data
mid = pd.read_csv(
    "data/sampled/common_microbe_disease_idx.dat",
    sep="\t",
    encoding="utf-8",
    header=None,
    names=["MicrobeIdx", "DiseaseIdx", "Weight"],
)
mime = pd.read_csv(
    "data/sampled/common_microbe_metabolite_idx.dat",
    sep="\t",
    encoding="utf-8",
    header=None,
    names=["MicrobeIdx", "MetaboliteIdx", "Weight"],
)
med = pd.read_csv(
    "data/sampled/common_metabolite_disease_idx.dat",
    sep="\t",
    encoding="utf-8",
    header=None,
    names=["MetaboliteIdx", "DiseaseIdx", "Weight"],
)

print(f"Number of Microbe-Disease edges: {len(mid)}")
print(f"Number of Microbe-Metabolite edges: {len(mime)}")
print(f"Number of Metabolite-Disease edges: {len(med)}")

# make directory if not exist
pathlib.Path("data/sampled/preprocessed/").mkdir(parents=True, exist_ok=True)
# microbe-disease
md_train, md_val, md_test = split_date(mid, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
save_split_data2npz(md_train, md_val, md_test, "data/sampled/preprocessed/micro_disease_train_val_test_idx.npz")

# training: 70%, validation: 20%, testing: 10%
train_val_test_idx = np.load("data/sampled/preprocessed/micro_disease_train_val_test_idx.npz")
train_idx = train_val_test_idx["train"]
val_idx = train_val_test_idx["val"]
test_idx = train_val_test_idx["test"]

# reset microbe-disease index
microbe_disease = mid.loc[train_idx].reset_index(drop=True)
print(f"Length of MID Training data: {len(microbe_disease)}")

# microbe-metabolite
mm_train, mm_val, mm_test = split_date(mime, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
save_split_data2npz(mm_train, mm_val, mm_test, "data/sampled/preprocessed/micro_metabolite_train_val_test_idx.npz")

# microbe-metabolite
train_val_test_idx_mm = np.load("data/sampled/preprocessed/micro_metabolite_train_val_test_idx.npz")
train_idx_mm = train_val_test_idx_mm["train"]
val_idx_mm = train_val_test_idx_mm["val"]
test_idx_mm = train_val_test_idx_mm["test"]

# reset microbe-metabolite index
microbe_metabolite = mime.loc[train_idx_mm].reset_index(drop=True)
print(f"Length of MIME Training data: {len(microbe_metabolite)}")


# hardcoded node num
save_prefix = "data/sampled/preprocessed/"

num_microbe = (mime["MicrobeIdx"].max() + 1).astype(np.int16)
num_disease = (med["DiseaseIdx"].max() + 1).astype(np.int16)
num_metabolite = (med["MetaboliteIdx"].max() + 1).astype(np.int16)

# build adjacency matrix
# 0 for microbe, 1 for disease, 2 for metabolite
dim = num_microbe + num_disease + num_metabolite

type_mask = np.zeros(dim, dtype=np.int16)
type_mask[num_microbe : num_microbe + num_disease] = 1
type_mask[num_microbe + num_disease :] = 2

adjM = np.zeros((dim, dim), dtype=np.int16)
for _, row in mid.iterrows():
    microID = row["MicrobeIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    adjM[microID, diseaseID] = 1
    adjM[diseaseID, microID] = 1
for _, row in mime.iterrows():
    microID = row["MicrobeIdx"]
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    adjM[microID, metID] = 1
    adjM[metID, microID] = 1
for _, row in med.iterrows():
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
    scaling_factor=1.0,
)

micro_d_meta_d_micro = lexicographical_sort(micro_d_meta_d_micro, [0, 2, 1, 3, 4])

# 0-2-1-2-0 (microbe-metabolite-disease-metabolite-microbe)
micro_meta_d_meta_micro = generate_long_relationship_array(
    relational_list=metabolite_microbe_list,
    intermediate_triplet=metabolite_disease_metabolite,
    num_offset1=(num_microbe + num_disease),
    scaling_factor=1.0,
)

micro_meta_d_meta_micro = lexicographical_sort(micro_meta_d_meta_micro, [0, 2, 1, 3, 4])

# 1-2-0-2-1 (disease-metabolite-microbe-metabolite-disease)
d_meta_micro_meta_d = generate_long_relationship_array(
    relational_list=metabolite_disease_list,
    intermediate_triplet=metabolite_microbe_metabolite,
    num_offset1=(num_microbe + num_disease),
    num_offset2=num_microbe,
    scaling_factor=1.0,
)

d_meta_micro_meta_d = lexicographical_sort(d_meta_micro_meta_d, [0, 2, 1, 3, 4])

# 2-1-0-1-2 (metabolite-disease-microbe-disease-metabolite)
meta_d_micro_d_meta = generate_long_relationship_array(
    relational_list=disease_metabolite_list,
    intermediate_triplet=disease_microbe_disease,
    num_offset1=num_microbe,
    num_offset2=(num_microbe + num_disease),
    scaling_factor=1.0,
)

meta_d_micro_d_meta = lexicographical_sort(meta_d_micro_d_meta, [0, 2, 1, 3, 4])

# 2-0-1-0-2 (metabolite-microbe-disease-microbe-metabolite)
meta_micro_d_micro_meta = generate_long_relationship_array(
    relational_list=microbe_metabolite_list,
    intermediate_triplet=microbe_disease_microbe,
    num_offset2=(num_microbe + num_disease),
    scaling_factor=1.0,
)

meta_micro_d_micro_meta = lexicographical_sort(meta_micro_d_micro_meta, [0, 2, 1, 3, 4])


expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 2, 0), (0, 2, 1, 2, 0)],
    [(1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1)],
    [(2, 0, 2), (2, 0, 1, 0, 2), (2, 1, 0, 1, 2), (2, 1, 2)],
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
    (2, 0, 2): metabolite_microbe_metabolite,
    (2, 0, 1, 0, 2): meta_micro_d_micro_meta,
    (2, 1, 0, 1, 2): meta_d_micro_d_meta,
    (2, 1, 2): metabolite_disease_metabolite,
}

# write all things
target_idx_lists = [np.arange(num_microbe), np.arange(num_disease), np.arange(num_metabolite)]
offset_list = [0, num_microbe, num_microbe + num_disease]
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

# output positive and negative samples for microbe-disease training, validation and testing
np.random.seed(453289)
save_prefix = "data/sampled/preprocessed/"
num_microbe = (mime["MicrobeIdx"].max() + 1).astype(np.int16)
num_disease = (med["DiseaseIdx"].max() + 1).astype(np.int16)
microbe_disease = np.load("data/sampled/preprocessed/microbe_disease.npy")
train_val_test_idx = np.load("data/sampled/preprocessed/micro_disease_train_val_test_idx.npz")
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

# # balance training negatives by sampling to match the number of positives
# train_neg_sampled = np.random.choice(
#     len(train_neg_candidates),
#     size=len(train_microbe_disease),  # match the number of positives
#     replace=False,
# )
# train_neg_candidates = train_neg_candidates[train_neg_sampled]

np.savez(
    save_prefix + "train_val_test_neg_microbe_disease.npz",
    train_neg_micro_dis=train_neg_candidates,
    val_neg_micro_dis=val_neg_candidates,
    test_neg_micro_dis=test_neg_candidates,
)
np.savez(
    save_prefix + "train_val_test_pos_microbe_disease.npz",
    train_pos_micro_dis=microbe_disease[train_idx],
    val_pos_micro_dis=microbe_disease[val_idx],
    test_pos_micro_dis=microbe_disease[test_idx],
)


# output microbe_metabolite.npy
microbe_metabolite = pd.read_csv(
    "data/sampled/microbe_metabolite.dat",
    encoding="utf-8",
    delimiter="\t",
    names=["MicrobeID", "MetaboliteID"],
)
microbe_metabolite = microbe_metabolite[["MicrobeID", "MetaboliteID"]].to_numpy()
np.save(save_prefix + "microbe_metabolite.npy", microbe_metabolite)

# output positive and negative samples for microbe-metabolite training, validation and testing
np.random.seed(453289)
save_prefix = "data/sampled/preprocessed/"
num_microbe = (mime["MicrobeIdx"].max() + 1).astype(np.int16)
num_metabolite = (med["MetaboliteIdx"].max() + 1).astype(np.int16)
microbe_metabolite = np.load("data/sampled/preprocessed/microbe_metabolite.npy")
train_val_test_idx = np.load("data/sampled/preprocessed/micro_metabolite_train_val_test_idx.npz")
train_idx = train_val_test_idx["train"]
val_idx = train_val_test_idx["val"]
test_idx = train_val_test_idx["test"]

neg_candidates = []
counter = 0
for i in range(num_microbe):
    for j in range(num_metabolite):
        if counter < len(microbe_metabolite):
            if i == microbe_metabolite[counter, 0] and j == microbe_metabolite[counter, 1]:
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)

idx = np.random.choice(len(neg_candidates), len(val_idx) + len(test_idx), replace=False)
val_neg_candidates = neg_candidates[sorted(idx[: len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx) :])]

train_microbe_metabolite = microbe_metabolite[train_idx]
train_neg_candidates = []
counter = 0
for i in range(num_microbe):
    for j in range(num_metabolite):
        if counter < len(train_microbe_metabolite):
            if i == train_microbe_metabolite[counter, 0] and j == train_microbe_metabolite[counter, 1]:
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

# balance training negatives by sampling to match the number of positives
train_neg_sampled = np.random.choice(
    len(train_neg_candidates),
    size=len(train_microbe_metabolite),  # match the number of positives
    replace=False,
)
train_neg_candidates = train_neg_candidates[train_neg_sampled]

np.savez(
    save_prefix + "train_val_test_neg_microbe_metabolite.npz",
    train_neg_micro_meta=train_neg_candidates,
    val_neg_micro_meta=val_neg_candidates,
    test_neg_micro_meta=test_neg_candidates,
)
np.savez(
    save_prefix + "train_val_test_pos_microbe_metabolite.npz",
    train_pos_micro_meta=microbe_metabolite[train_idx],
    val_pos_micro_meta=microbe_metabolite[val_idx],
    test_pos_micro_meta=microbe_metabolite[test_idx],
)
