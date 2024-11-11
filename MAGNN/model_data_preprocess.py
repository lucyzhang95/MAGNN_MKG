import cProfile
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import scipy
from MAGNN_preprocess_utils.preprocess import save_split_data2npz, split_date

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

# reset microbe-disease index
microbe_disease = microbe_disease.loc[train_idx].reset_index(drop=True)
microbe_disease.head()
print(f"Length of Training data: {len(microbe_disease)}")

# build adjacency matrix
# 0 for microbe, 1 for disease, 2 for metabolite
dim = num_microbe + num_disease + num_metabolite

type_mask = np.zeros(dim, dtype=int)
type_mask[num_microbe : num_microbe + num_disease] = 1
type_mask[num_microbe + num_disease :] = 2

adjM = np.zeros((dim, dim), dtype=int)
for _, row in microbe_disease.iterrows():
    microID = row["MicrobeIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    # increment accounts for multiple links exist between same microbe and disease relationships
    adjM[microID, diseaseID] += 1
    adjM[diseaseID, microID] += 1
for _, row in microbe_metabolite.iterrows():
    microID = row["MicrobeIdx"]
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    adjM[microID, metID] += 1
    adjM[metID, microID] += 1
for _, row in metabolite_disease.iterrows():
    metID = num_microbe + num_disease + row["MetaboliteIdx"]
    diseaseID = num_microbe + row["DiseaseIdx"]
    adjM[metID, diseaseID] = 1
    adjM[diseaseID, metID] = 1

# map each microbe to a list of diseases
microbe_disease_list = {
    i: adjM[i, num_microbe : num_microbe + num_disease].nonzero()[0]
    for i in range(num_microbe)
}
# map each disease to a list of microbes
disease_microbe_list = {
    i: adjM[num_microbe + i, :num_microbe].nonzero()[0]
    for i in range(num_disease)
}
# map each metabolite to a list of diseases
metabolite_disease_list = {
    i: adjM[
        num_microbe + num_disease + i, num_microbe : num_microbe + num_disease
    ].nonzero()[0]
    for i in range(num_metabolite)
}
# map each disease to a list of metabolites
disease_metabolite_list = {
    i: adjM[
        num_microbe + i,
        num_microbe + num_disease : num_microbe + num_disease + num_metabolite,
    ].nonzero()[0]
    for i in range(num_disease)
}
# map each microbe to a list of metabolites
microbe_metabolite_list = {
    i: adjM[
        i,
        num_microbe + num_disease : num_microbe + num_disease + num_metabolite,
    ].nonzero()[0]
    for i in range(num_microbe)
}
# map each metabolite to a list of microbes
metabolite_microbe_list = {
    i: adjM[num_microbe + num_disease + i, :num_microbe].nonzero()[0]
    for i in range(num_metabolite)
}


edge_metapath_idx_profiler1 = cProfile.Profile()
edge_metapath_idx_profiler1.enable()

# 0-1-0 (microbe-disease-microbe)
microbe_disease_microbe = np.array(
    [
        (microbe1, disease, microbe2)
        for disease, microbe_list in disease_microbe_list.items()
        for microbe1 in microbe_list
        for microbe2 in microbe_list
    ],
    dtype=np.int32,
)
microbe_disease_microbe[:, 1] += num_microbe

# sort by order [0, 2, 1]
microbe_disease_microbe = microbe_disease_microbe[
    np.lexsort(
        (
            microbe_disease_microbe[:, 1],
            microbe_disease_microbe[:, 2],
            microbe_disease_microbe[:, 0],
        )
    )
]

edge_metapath_idx_profiler1.disable()
print("Profile for microbe-disease-microbe edge_metapath_idx_profiler:")
edge_metapath_idx_profiler1.print_stats()

# 0-2-0 (microbe-metabolite-microbe)
microbe_metabolite_microbe = np.array(
    [
        (microbe1, metabolite, microbe2)
        for metabolite, microbe_list in metabolite_microbe_list.items()
        for microbe1 in microbe_list
        for microbe2 in microbe_list
    ],
    dtype=np.int32,
)
microbe_metabolite_microbe[:, 1] += num_microbe + num_disease

microbe_metabolite_microbe = microbe_metabolite_microbe[
    np.lexsort(
        (
            microbe_metabolite_microbe[:, 1],
            microbe_metabolite_microbe[:, 2],
            microbe_metabolite_microbe[:, 0],
        )
    )
]

# 1-2-1 (disease-metabolite-disease)
disease_metabolite_disease = np.array(
    [
        (d1, metabolite, d2)
        for metabolite, disease_list in metabolite_disease_list.items()
        for d1 in disease_list
        for d2 in disease_list
    ],
    dtype=np.int32,
)
disease_metabolite_disease[:, [0, 2]] += num_microbe
disease_metabolite_disease[:, 1] += num_disease

disease_metabolite_disease = disease_metabolite_disease[
    np.lexsort(
        (
            disease_metabolite_disease[:, 1],
            disease_metabolite_disease[:, 2],
            disease_metabolite_disease[:, 0],
        )
    )
]

# 2-1-2 (metabolite-disease-metabolite)
metabolite_disease_metabolite = np.array(
    [
        (m1, disease, m2)
        for disease, metabolite_list in disease_metabolite_list.items()
        for m1 in metabolite_list
        for m2 in metabolite_list
    ],
    dtype=np.int32,
)
metabolite_disease_metabolite[:, [0, 2]] += num_microbe + num_disease
metabolite_disease_metabolite[:, 1] -= num_disease

metabolite_disease_metabolite = metabolite_disease_metabolite[
    np.lexsort(
        (
            metabolite_disease_metabolite[:, 1],
            metabolite_disease_metabolite[:, 2],
            metabolite_disease_metabolite[:, 0],
        )
    )
]


edge_metapath_idx_profiler2 = cProfile.Profile()
edge_metapath_idx_profiler1.enable()

# 0-1-2-1-0 (microbe-disease-metabolite-disease-microbe)
micro_d_meta_d_micro = []
for d1, meta, d2 in disease_metabolite_disease:
    if (
        len(disease_microbe_list[d1 - num_microbe]) == 0
        or len(disease_microbe_list[d2 - num_microbe]) == 0
    ):
        continue

    candidate_microbe1_list = np.random.choice(
        disease_microbe_list[d1 - num_microbe],
        int(0.2 * len(disease_microbe_list[d1 - num_microbe])),
        replace=False,
    )
    candidate_microbe2_list = np.random.choice(
        disease_microbe_list[d2 - num_microbe],
        int(0.2 * len(disease_microbe_list[d2 - num_microbe])),
        replace=False,
    )

    micro_d_meta_d_micro.extend(
        (microbe1, d1, meta, d2, microbe2)
        for microbe1 in candidate_microbe1_list
        for microbe2 in candidate_microbe2_list
    )

micro_d_meta_d_micro = np.array(micro_d_meta_d_micro, dtype=np.int32)
# sort by order [0, 4, 1, 2, 3]
micro_d_meta_d_micro = micro_d_meta_d_micro[
    np.lexsort(
        (
            micro_d_meta_d_micro[:, 3],
            micro_d_meta_d_micro[:, 2],
            micro_d_meta_d_micro[:, 1],
            micro_d_meta_d_micro[:, 4],
            micro_d_meta_d_micro[:, 0],
        )
    )
]

edge_metapath_idx_profiler2.disable()
print(
    "Profile for microbe-disease-metabolite-disease-microbe edge_metapath_idx_profiler:"
)
edge_metapath_idx_profiler2.print_stats()

# 1-0-2-0-1 (disease-microbe-metabolite-microbe-disease)
d_micro_meta_micro_d = []
for micro1, meta, micro2 in microbe_metabolite_microbe:
    if (
        len(microbe_disease_list[micro1]) == 0
        or len(microbe_disease_list[micro2]) == 0
    ):
        continue

    candidate_d1_list = np.random.choice(
        microbe_disease_list[micro1],
        int(0.2 * len(microbe_disease_list[micro1])),
        replace=False,
    )
    candidate_d2_list = np.random.choice(
        microbe_disease_list[micro2],
        int(0.2 * len(microbe_disease_list[micro2])),
        replace=False,
    )

    d_micro_meta_micro_d.extend(
        (d1, micro1, meta, micro2, d2)
        for d1 in candidate_d1_list
        for d2 in candidate_d2_list
    )

d_micro_meta_micro_d = np.array(d_micro_meta_micro_d, dtype=np.int32)
d_micro_meta_micro_d = d_micro_meta_micro_d[
    np.lexsort(
        (
            d_micro_meta_micro_d[:, 3],
            d_micro_meta_micro_d[:, 2],
            d_micro_meta_micro_d[:, 1],
            d_micro_meta_micro_d[:, 4],
            d_micro_meta_micro_d[:, 0],
        )
    )
]

# 0-2-1-2-0 (microbe-metabolite-disease-metabolite-microbe)
micro_meta_d_meta_micro = []
for meta1, d, meta2 in metabolite_disease_metabolite:
    if (
        len(metabolite_microbe_list[meta1 - num_microbe - num_disease]) == 0
        or len(metabolite_microbe_list[meta2 - num_microbe - num_disease]) == 0
    ):
        continue

    candidate_micro1_list = np.random.choice(
        metabolite_microbe_list[meta1 - num_microbe - num_disease],
        int(
            0.2
            * len(metabolite_microbe_list[meta1 - num_microbe - num_disease])
        ),
        replace=False,
    )
    candidate_micro2_list = np.random.choice(
        metabolite_microbe_list[meta2 - num_microbe - num_disease],
        int(
            0.2
            * len(metabolite_microbe_list[meta2 - num_microbe - num_disease])
        ),
        replace=False,
    )

    micro_meta_d_meta_micro.extend(
        (micro1, meta1, d, meta2, micro2)
        for micro1 in candidate_micro1_list
        for micro2 in candidate_micro2_list
    )

micro_meta_d_meta_micro = np.array(micro_meta_d_meta_micro, dtype=np.int32)
micro_meta_d_meta_micro = micro_meta_d_meta_micro[
    np.lexsort(
        (
            micro_meta_d_meta_micro[:, 3],
            micro_meta_d_meta_micro[:, 2],
            micro_meta_d_meta_micro[:, 1],
            micro_meta_d_meta_micro[:, 4],
            micro_meta_d_meta_micro[:, 0],
        )
    )
]

# 1-0-1 (disease-microbe-disease)
disease_microbe_disease = np.array(
    [
        (d1, microbe, d2)
        for microbe, disease_list in microbe_disease_list.items()
        for d1 in disease_list
        for d2 in disease_list
    ],
    dtype=np.int32,
)
disease_microbe_disease[:, [0, 2]] += num_microbe

disease_microbe_disease = disease_microbe_disease[
    np.lexsort(
        (
            disease_microbe_disease[:, 1],
            disease_microbe_disease[:, 2],
            disease_microbe_disease[:, 0],
        )
    )
]

# 2-0-2 (metabolite-microbe-metabolite)
metabolite_microbe_metabolite = np.array(
    [
        (meta1, microbe, meta2)
        for microbe, metabolite_list in microbe_metabolite_list.items()
        for meta1 in metabolite_list
        for meta2 in metabolite_list
    ],
    dtype=np.int32,
)
metabolite_microbe_metabolite[:, [0, 2]] += num_microbe + num_disease

metabolite_microbe_metabolite = metabolite_microbe_metabolite[
    np.lexsort(
        (
            metabolite_microbe_metabolite[:, 1],
            metabolite_microbe_metabolite[:, 2],
            metabolite_microbe_metabolite[:, 0],
        )
    )
]

# 1-2-0-2-1 (disease-metabolite-microbe-metabolite-disease)
d_meta_micro_meta_d = []
for meta1, micro, meta2 in metabolite_microbe_metabolite.items():
    if (
        len(metabolite_disease_list[meta1 - num_microbe - num_disease]) == 0
        or len(metabolite_disease_list[meta2 - num_microbe - num_disease]) == 0
    ):
        continue

    candidate_d1_list = np.random.choice(
        metabolite_disease_list[meta1 - num_microbe - num_disease],
        int(
            0.2
            * len(metabolite_disease_list[meta1 - num_microbe - num_disease])
        ),
        replace=False,
    )
    candidate_d2_list = np.random.choice(
        metabolite_disease_list[meta2 - num_microbe - num_disease],
        int(
            0.2
            * len(metabolite_disease_list[meta2 - num_microbe - num_disease])
        ),
        replace=False,
    )

    d_meta_micro_meta_d.extend(
        (d1, meta1, micro, meta2, d2)
        for d1 in candidate_d1_list
        for d2 in candidate_d2_list
    )

d_meta_micro_meta_d = np.array(d_meta_micro_meta_d, dtype=np.int32)

d_meta_micro_meta_d = d_meta_micro_meta_d[
    np.lexsort(
        (
            d_meta_micro_meta_d[:, 3],
            d_meta_micro_meta_d[:, 2],
            d_meta_micro_meta_d[:, 1],
            d_meta_micro_meta_d[:, 4],
            d_meta_micro_meta_d[:, 0],
        )
    )
]

# 2-0-1-0-2 (metabolite-microbe-disease-microbe-metabolite)
meta_micro_d_micro_meta = []
for micro1, d, micro2 in microbe_metabolite_microbe.items():
    if (
        len(microbe_metabolite_list[micro1]) == 0
        or len(microbe_metabolite_list[micro2]) == 0
    ):
        continue

    candidate_meta1_list = np.random.choice(
        microbe_metabolite_list[micro1],
        int(0.2 * len(microbe_metabolite_list[micro1])),
        replace=False,
    )
    candidate_meta2_list = np.random.choice(
        microbe_metabolite_list[micro2],
        int(0.2 * len(microbe_metabolite_list[micro2])),
        replace=False,
    )

    meta_micro_d_micro_meta.extend(
        (meta1, micro1, d, micro2, meta2)
        for meta1 in candidate_meta1_list
        for meta2 in candidate_meta2_list
    )

meta_micro_d_micro_meta = np.array(meta_micro_d_micro_meta, dtype=np.int32)

meta_micro_d_micro_meta = meta_micro_d_micro_meta[
    np.lexsort(
        (
            meta_micro_d_micro_meta[:, 3],
            meta_micro_d_micro_meta[:, 2],
            meta_micro_d_micro_meta[:, 1],
            meta_micro_d_micro_meta[:, 4],
            meta_micro_d_micro_meta[:, 0],
        )
    )
]

# 2-1-0-1-2 (metabolite-disease-microbe-disease-metabolite)
meta_d_micro_d_meta = []
for d1, micro, d2 in disease_microbe_disease.items():
    if (
        len(disease_metabolite_list[d1 - num_microbe]) == 0
        or len(disease_metabolite_list[d2 - num_microbe]) == 0
    ):
        continue

    candidate_meta1_list = np.random.choice(
        disease_metabolite_list[d1 - num_microbe],
        int(0.2 * len(disease_metabolite_list[d1 - num_microbe])),
        replace=False,
    )
    candidate_meta2_list = np.random.choice(
        disease_metabolite_list[d2 - num_microbe],
        int(0.2 * len(disease_metabolite_list[d2 - num_microbe])),
        replace=False,
    )

    meta_d_micro_d_meta.extend(
        (meta1, d1, micro, d2, meta2)
        for meta1 in candidate_meta1_list
        for meta2 in candidate_meta2_list
    )

meta_d_micro_d_meta = np.array(meta_d_micro_d_meta, dtype=np.int32)

meta_d_micro_d_meta = meta_d_micro_d_meta[
    np.lexsort(
        (
            meta_d_micro_d_meta[:, 3],
            meta_d_micro_d_meta[:, 2],
            meta_d_micro_d_meta[:, 1],
            meta_d_micro_d_meta[:, 4],
            meta_d_micro_d_meta[:, 0],
        )
    )
]

metapath_idx_mapping_profile = cProfile.Profile()
metapath_idx_mapping_profile.enable()

expected_metapaths = [
    (0, 1, 0),
    (0, 2, 0),
    (1, 0, 1),
    (2, 0, 2),
    (1, 2, 1),
    (2, 1, 2),
    (0, 1, 2, 1, 0),
    (1, 0, 2, 0, 1),
    (0, 2, 1, 2, 0),
    (1, 2, 0, 2, 1),
    (2 - 0 - 1 - 0 - 2),
    (2 - 1 - 0 - 1 - 2),
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + "{}".format(i)).mkdir(
        parents=True, exist_ok=True
    )

metapath_indices_mapping = {
    (0, 1, 0): microbe_disease_microbe,
    (0, 2, 0): microbe_metabolite_microbe,
    (1, 0, 1): disease_microbe_disease,
    (2, 0, 2): metabolite_microbe_metabolite,
    (1, 2, 1): disease_metabolite_disease,
    (2, 1, 2): metabolite_disease_metabolite,
    (0, 1, 2, 1, 0): micro_d_meta_d_micro,
    (1, 0, 2, 0, 1): d_micro_meta_micro_d,
    (0, 2, 1, 2, 0): micro_meta_d_meta_micro,
    (1, 2, 0, 2, 1): d_meta_micro_meta_d,
    (2, 0, 1, 0, 2): meta_micro_d_micro_meta,
    (2, 1, 0, 1, 2): meta_d_micro_d_meta,
}

metapath_idx_mapping_profile.disable()
print("Profile for metapath_idx_mapping_profile:")
metapath_idx_mapping_profile.print_stats()

# write all things
target_idx_lists = [np.arange(num_microbe), np.arange(num_disease)]
offset_list = [0, num_microbe]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(
            save_prefix
            + "{}/".format(i)
            + "-".join(map(str, metapath))
            + "_idx.pickle",
            "wb",
        ) as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while (
                    right < len(edge_metapath_idx_array)
                    and edge_metapath_idx_array[right, 0]
                    == target_idx + offset_list[i]
                ):
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[
                    left:right, ::-1
                ]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        # np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        with open(
            save_prefix
            + "{}/".format(i)
            + "-".join(map(str, metapath))
            + ".adjlist",
            "w",
        ) as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while (
                    right < len(edge_metapath_idx_array)
                    and edge_metapath_idx_array[right, 0]
                    == target_idx + offset_list[i]
                ):
                    right += 1
                neighbors = (
                    edge_metapath_idx_array[left:right, -1] - offset_list[i]
                )
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write(
                        "{} ".format(target_idx) + " ".join(neighbors) + "\n"
                    )
                else:
                    out_file.write("{}\n".format(target_idx))
                left = right

# save scipy sparse adjM
scipy.sparse.save_npz(save_prefix + "adjM.npz", scipy.sparse.csr_matrix(adjM))
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
        if counter < len(num_disease):
            if i == num_disease[counter, 0] and j == num_disease[counter, 1]:
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
