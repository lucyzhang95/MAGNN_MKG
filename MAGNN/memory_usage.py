import numpy as np
import pandas as pd
from memory_profiler import profile


@profile
def build_adj_matrix():
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
    return adjM


def generate_edge_metapath_idx(adjM):
    num_microbe = 8202
    num_disease = 898

    # map each disease to a list of microbes
    disease_microbe_list = {
        i: adjM[num_microbe + i, :num_microbe].nonzero()[0]
        for i in range(num_disease)
    }

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

    microbe_disease_microbe = microbe_disease_microbe[
        np.lexsort(
            (
                microbe_disease_microbe[:, 1],
                microbe_disease_microbe[:, 2],
                microbe_disease_microbe[:, 0],
            )
        )
    ]
    return microbe_disease_microbe


if __name__ == "__main__":
    adjM = build_adj_matrix()
    generate_edge_metapath_idx(adjM)
