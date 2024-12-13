import time

import numpy as np
import pandas as pd
from MAGNN_preprocess_utils.preprocess import (
    generate_long_relationship_array,
    generate_triplet_array,
    lexicographical_sort,
    process_single_metapath_in_batches_to_single_file,
    save_split_data2npz,
    split_date,
)
from scipy.sparse import csr_matrix, save_npz

start_time = time.time()

save_prefix = "data/preprocessed/"

microbe_disease = pd.read_csv(
    "data/raw/microbe_disease_idx.dat",
    encoding="utf-8",
    delimiter="\t",
    names=["MicrobeIdx", "DiseaseIdx"],
)
microbe_metabolite = pd.read_csv(
    "data/raw/microbe_metabolite_idx.dat",
    encoding="utf-8",
    delimiter="\t",
    names=["MicrobeIdx", "MetaboliteIdx"],
)
metabolite_disease = pd.read_csv(
    "data/raw/metabolite_disease_idx.dat",
    encoding="utf-8",
    delimiter="\t",
    names=["MetaboliteIdx", "DiseaseIdx"],
)
num_microbe = 8202
num_metabolite = 23823
num_disease = 898

md_train, md_val, md_test = split_date(
    microbe_disease, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)
save_split_data2npz(
    md_train, md_val, md_test, "data/micro_disease_train_val_test_idx.npz"
)

# training: 70%, validation: 20%, testing: 10%
train_val_test_idx = np.load("data/raw/micro_disease_train_val_test_idx.npz")
train_idx = train_val_test_idx["train"]
val_idx = train_val_test_idx["val"]
test_idx = train_val_test_idx["test"]

microbe_disease = microbe_disease.loc[train_idx].reset_index(drop=True)
print(f"Length of Training data: {len(microbe_disease)}")

# build adjacency matrix
# 0 for microbe, 1 for disease, 2 for metabolite
dim = num_microbe + num_disease + num_metabolite

type_mask = np.zeros(dim, dtype=np.int16)
type_mask[num_microbe : num_microbe + num_disease] = 1
type_mask[num_microbe + num_disease :] = 2

# did not do increments here
adjM = np.zeros((dim, dim), dtype=np.int16)
for _, row in microbe_disease.iterrows():
    microID = row["MicrobeIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    adjM[microID, diseaseID] = 1
    adjM[diseaseID, microID] = 1
for _, row in microbe_metabolite.iterrows():
    microID = row["MicrobeIdx"]
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    adjM[microID, metID] = 1
    adjM[metID, microID] = 1
for _, row in metabolite_disease.iterrows():
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    adjM[metID, diseaseID] = 1
    adjM[diseaseID, metID] = 1

# map each microbe to a list of diseases within adjM and remove empty arrays
# adjM[microbe, diseases]
microbe_disease_list = {
    i: adjM[i, num_microbe : num_microbe + num_disease]
    .nonzero()[0]
    .astype(np.int16)
    for i in range(num_microbe)
}
microbe_disease_list = {
    i: v for i, v in microbe_disease_list.items() if v.size > 0
}

# map each disease to a list of microbes within adjM and remove empty arrays
# adjM[disease, microbes]
disease_microbe_list = {
    i: adjM[num_microbe + i, :num_microbe].nonzero()[0].astype(np.int16)
    for i in range(num_disease)
}
disease_microbe_list = {
    i: v for i, v in disease_microbe_list.items() if v.size > 0
}

# map each metabolite to a list of diseases within adjM and remove empty arrays
# adjM[metabolite, diseases]
metabolite_disease_list = {
    i: adjM[
        num_microbe + num_disease + i, num_microbe : num_microbe + num_disease
    ]
    .nonzero()[0]
    .astype(np.int16)
    for i in range(num_metabolite)
}
metabolite_disease_list = {
    i: v for i, v in metabolite_disease_list.items() if v.size > 0
}

# map each disease to a list of metabolites within adjM and remove empty arrays
# adjM[disease, metabolites]
disease_metabolite_list = {
    i: adjM[num_microbe + i, num_microbe + num_disease :]
    .nonzero()[0]
    .astype(np.int16)
    for i in range(num_disease)
}
disease_metabolite_list = {
    i: v for i, v in disease_metabolite_list.items() if v.size > 0
}

# map each microbe to a list of metabolites within adjM and remove empty arrays
# adjM[microbe, metabolites]
microbe_metabolite_list = {
    i: adjM[i, num_microbe + num_disease :].nonzero()[0].astype(np.int16)
    for i in range(num_microbe)
}
microbe_metabolite_list = {
    i: v for i, v in microbe_metabolite_list.items() if v.size > 0
}

# map each metabolite to a list of microbes within adjM and remove empty arrays
# adjM[metabolite, microbes]
metabolite_microbe_list = {
    i: adjM[num_microbe + num_disease + i, :num_microbe]
    .nonzero()[0]
    .astype(np.int16)
    for i in range(num_metabolite)
}
metabolite_microbe_list = {
    i: v for i, v in metabolite_microbe_list.items() if v.size > 0
}

# 0-1-0 (microbe-disease-microbe)
# remove the same metapath types with reverse order. e.g., (1, 0, 2) and (2, 0, 1) are the same
# remove path includes the same microbe1 and microbe2 (same 1st and last element). e.g., (1, 4, 1) and (0, 4, 0) are removed
microbe_disease_microbe = generate_triplet_array(disease_microbe_list)
microbe_disease_microbe[:, 1] += num_microbe
microbe_disease_microbe = lexicographical_sort(
    microbe_disease_microbe, [0, 2, 1]
)

# save 0-1-0 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(0, 1, 0),
    metapath_array=microbe_disease_microbe,
    target_idx_list=np.arange(num_microbe),
    offset=0,
    save_prefix=save_prefix,
    group_index=0,
)

# 2-0-1-0-2 (metabolite-microbe-disease-microbe-metabolite)
meta_micro_d_micro_meta = generate_long_relationship_array(
    relational_list=microbe_metabolite_list,
    intermediate_triplet=microbe_disease_microbe,
    num_offset2=(num_microbe + num_disease),
)

meta_micro_d_micro_meta = lexicographical_sort(
    meta_micro_d_micro_meta, [0, 2, 1, 3, 4]
)

# save 2-0-1-0-2 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(2, 0, 1, 0, 2),
    metapath_array=meta_micro_d_micro_meta,
    target_idx_list=np.arange(num_metabolite),
    offset=num_microbe + num_disease,
    save_prefix=save_prefix,
    group_index=2,
)

del microbe_disease_microbe
del meta_micro_d_micro_meta

# 0-2-0 (microbe-metabolite-microbe)
microbe_metabolite_microbe = generate_triplet_array(metabolite_microbe_list)
microbe_metabolite_microbe[:, 1] += num_microbe + num_disease
microbe_metabolite_microbe = lexicographical_sort(
    microbe_metabolite_microbe, [0, 2, 1]
)

# save 0-2-0 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(0, 2, 0),
    metapath_array=microbe_metabolite_microbe,
    target_idx_list=np.arange(num_microbe),
    offset=0,
    save_prefix=save_prefix,
    group_index=0,
)

# 1-0-2-0-1 (disease-microbe-metabolite-microbe-disease)
d_micro_meta_micro_d = generate_long_relationship_array(
    relational_list=microbe_disease_list,
    intermediate_triplet=microbe_metabolite_microbe,
    num_offset2=num_microbe,
)

d_micro_meta_micro_d = lexicographical_sort(
    d_micro_meta_micro_d, [0, 2, 1, 3, 4]
)

# save 1-0-2-0-1 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(1, 0, 2, 0, 1),
    metapath_array=d_micro_meta_micro_d,
    target_idx_list=np.arange(num_disease),
    offset=num_microbe,
    save_prefix=save_prefix,
    group_index=1,
)

del microbe_metabolite_microbe
del d_micro_meta_micro_d

# 1-2-1 (disease-metabolite-disease)
disease_metabolite_disease = generate_triplet_array(metabolite_disease_list)
disease_metabolite_disease[:, (0, 2)] += num_microbe
disease_metabolite_disease[:, 1] += num_microbe + num_disease
disease_metabolite_disease = lexicographical_sort(
    disease_metabolite_disease, [0, 2, 1]
)

# save 1-2-1 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(1, 2, 1),
    metapath_array=disease_metabolite_disease,
    target_idx_list=np.arange(num_disease),
    offset=num_microbe,
    save_prefix=save_prefix,
    group_index=1,
)

# 0-1-2-1-0 (microbe-disease-metabolite-disease-microbe)
micro_d_meta_d_micro = generate_long_relationship_array(
    relational_list=disease_microbe_list,
    intermediate_triplet=disease_metabolite_disease,
    num_offset1=num_microbe,
)

micro_d_meta_d_micro = lexicographical_sort(
    micro_d_meta_d_micro, [0, 2, 1, 3, 4]
)

# save 0-1-2-1-0 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(0, 1, 2, 1, 0),
    metapath_array=micro_d_meta_d_micro,
    target_idx_list=np.arange(num_microbe),
    offset=0,
    save_prefix=save_prefix,
    group_index=0,
)

del disease_metabolite_disease
del micro_d_meta_d_micro

# 2-1-2 (metabolite-disease-metabolite)
metabolite_disease_metabolite = generate_triplet_array(disease_metabolite_list)
metabolite_disease_metabolite[:, (0, 2)] += num_microbe + num_disease
metabolite_disease_metabolite[:, 1] += num_microbe
metabolite_disease_metabolite = lexicographical_sort(
    metabolite_disease_metabolite, [0, 2, 1]
)

# save 2-1-2 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(2, 1, 2),
    metapath_array=metabolite_disease_metabolite,
    target_idx_list=np.arange(num_metabolite),
    offset=num_microbe + num_disease,
    save_prefix=save_prefix,
    group_index=2,
)

# 0-2-1-2-0 (microbe-metabolite-disease-metabolite-microbe)
micro_meta_d_meta_micro = generate_long_relationship_array(
    relational_list=metabolite_microbe_list,
    intermediate_triplet=metabolite_disease_metabolite,
    num_offset1=(num_microbe + num_disease),
)

micro_meta_d_meta_micro = lexicographical_sort(
    micro_meta_d_meta_micro, [0, 2, 1, 3, 4]
)

# save 0-2-1-2-0 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(0, 2, 1, 2, 0),
    metapath_array=micro_meta_d_meta_micro,
    target_idx_list=np.arange(num_microbe),
    offset=0,
    save_prefix=save_prefix,
    group_index=0,
)

del metabolite_disease_metabolite
del micro_meta_d_meta_micro

# 1-0-1 (disease-microbe-disease)
disease_microbe_disease = generate_triplet_array(microbe_disease_list)
disease_microbe_disease[:, (0, 2)] += num_microbe
disease_microbe_disease = lexicographical_sort(
    disease_microbe_disease, [0, 2, 1]
)

# save 1-0-1 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(1, 0, 1),
    metapath_array=disease_microbe_disease,
    target_idx_list=np.arange(num_disease),
    offset=num_microbe,
    save_prefix=save_prefix,
    group_index=1,
)

# 2-1-0-1-2 (metabolite-disease-microbe-disease-metabolite)
meta_d_micro_d_meta = generate_long_relationship_array(
    relational_list=disease_metabolite_list,
    intermediate_triplet=disease_microbe_disease,
    num_offset1=num_microbe,
    num_offset2=(num_microbe + num_disease),
)

meta_d_micro_d_meta = lexicographical_sort(
    meta_d_micro_d_meta, [0, 2, 1, 3, 4]
)

# save 2-1-0-1-2 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(2, 1, 0, 1, 2),
    metapath_array=meta_d_micro_d_meta,
    target_idx_list=np.arange(num_metabolite),
    offset=num_microbe + num_disease,
    save_prefix=save_prefix,
    group_index=2,
)

del disease_microbe_disease
del meta_d_micro_d_meta

# 2-0-2 (metabolite-microbe-metabolite)
metabolite_microbe_metabolite = generate_triplet_array(microbe_metabolite_list)
metabolite_microbe_metabolite[:, (0, 2)] += num_microbe + num_disease
metabolite_microbe_metabolite = lexicographical_sort(
    metabolite_microbe_metabolite, [0, 2, 1]
)

# save 2-0-2 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(2, 0, 2),
    metapath_array=metabolite_microbe_metabolite,
    target_idx_list=np.arange(num_metabolite),
    offset=num_microbe + num_disease,
    save_prefix=save_prefix,
    group_index=2,
)

# 1-2-0-2-1 (disease-metabolite-microbe-metabolite-disease)
d_meta_micro_meta_d = generate_long_relationship_array(
    relational_list=metabolite_disease_list,
    intermediate_triplet=metabolite_microbe_metabolite,
    num_offset1=(num_microbe + num_disease),
    num_offset2=num_microbe,
)

d_meta_micro_meta_d = lexicographical_sort(
    d_meta_micro_meta_d, [0, 2, 1, 3, 4]
)

# save 1-2-0-2-1 in batches
process_single_metapath_in_batches_to_single_file(
    metapath_type=(1, 2, 0, 2, 1),
    metapath_array=d_meta_micro_meta_d,
    target_idx_list=np.arange(num_disease),
    offset=num_microbe,
    save_prefix=save_prefix,
    group_index=1,
)

del metabolite_microbe_metabolite
del d_meta_micro_meta_d

# save scipy sparse adjM
save_npz(save_prefix + "adjM.npz", csr_matrix(adjM))
# save node type_mask
np.save(save_prefix + "node_types.npy", type_mask)

# output microbe_disease.npy
microbe_disease = pd.read_csv(
    "data/raw/microbe_disease_idx.dat",
    encoding="utf-8",
    delimiter="\t",
    names=["MicrobeID", "DiseaseID"],
)
microbe_disease = microbe_disease[["MicrobeID", "DiseaseID"]].to_numpy()
np.save(save_prefix + "microbe_disease.npy", microbe_disease)

# output positive and negative samples for training, validation and testing
np.random.seed(453289)
save_prefix = "data/preprocessed/microbe_disease_neg_pos_processed/"
num_microbe = 8202
num_disease = 898
microbe_disease = np.load("data/preprocessed/microbe_disease.npy")
train_val_test_idx = np.load("data/raw/micro_disease_train_val_test_idx.npz")
train_idx = train_val_test_idx["train"]
val_idx = train_val_test_idx["val"]
test_idx = train_val_test_idx["test"]

neg_candidates = []
counter = 0
for i in range(num_microbe):
    for j in range(num_disease):
        if counter < len(microbe_disease):
            if (
                i == microbe_disease[counter, 0]
                and j == microbe_disease[counter, 1]
            ):
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)

idx = np.random.choice(
    len(neg_candidates), len(val_idx) + len(test_idx), replace=False
)
val_neg_candidates = neg_candidates[sorted(idx[: len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx) :])]

train_microbe_disease = microbe_disease[train_idx]
train_neg_candidates = []
counter = 0
for i in range(num_microbe):
    for j in range(num_disease):
        if counter < len(train_microbe_disease):
            if (
                i == train_microbe_disease[counter, 0]
                and j == train_microbe_disease[counter, 1]
            ):
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

np.savez(
    save_prefix + "train_val_test_neg_microbe_disease.npz",
    train_neg_user_artist=train_neg_candidates,
    val_neg_user_artist=val_neg_candidates,
    test_neg_user_artist=test_neg_candidates,
)
np.savez(
    save_prefix + "train_val_test_pos_microbe_disease.npz",
    train_pos_user_artist=microbe_disease[train_idx],
    val_pos_user_artist=microbe_disease[val_idx],
    test_pos_user_artist=microbe_disease[test_idx],
)

end_time = time.time()
time_spent = end_time - start_time
print(f"The overall time spent is {time_spent} seconds.")
