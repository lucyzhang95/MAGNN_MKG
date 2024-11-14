import numpy as np
from memory_profiler import profile

# generate random datasets for testing
num_microbe = 70
num_disease = 20
num_metabolite = 10

dim = num_microbe + num_disease + num_metabolite

type_mask = np.zeros(dim, dtype=np.int16)
type_mask[num_microbe : num_microbe + num_disease] = 1
type_mask[num_microbe + num_disease :] = 2

adjM = np.random.choice([0, 1], size=(dim, dim)).astype(np.int16)

disease_microbe_list = {
    i: adjM[num_microbe + i, :num_microbe].nonzero()[0]
    for i in range(num_disease)
}


def generate_microbe_disease_microbe(disease_microbe_list, num_microbe):
    for disease, microbe_list in disease_microbe_list.items():
        # Adjust disease index based on num_microbe to map back to adjacency matrix
        disease_index = disease + num_microbe
        for microbe1 in microbe_list:
            for microbe2 in microbe_list:
                yield (microbe1, disease_index, microbe2)


@profile
def original_approach(disease_microbe_list, num_microbe):
    microbe_disease_microbe = []
    for disease, microbe_list in disease_microbe_list.items():
        microbe_disease_microbe.extend(
            [
                (microbe1, disease, microbe2)
                for microbe1 in microbe_list
                for microbe2 in microbe_list
            ]
        )
    microbe_disease_microbe = np.array(microbe_disease_microbe)
    microbe_disease_microbe[:, 1] += num_microbe
    sorted_index = sorted(
        list(range(len(microbe_disease_microbe))),
        key=lambda i: microbe_disease_microbe[i, [0, 2, 1]].tolist(),
    )
    microbe_disease_microbe = microbe_disease_microbe[sorted_index]
    return microbe_disease_microbe


@profile
def generator_approach(disease_microbe_list, num_microbe):
    microbe_disease_microbe_gen = generate_microbe_disease_microbe(
        disease_microbe_list, num_microbe
    )
    microbe_disease_microbe_sorted = sorted(
        microbe_disease_microbe_gen, key=lambda x: (x[0], x[2], x[1])
    )
    microbe_disease_microbe_array = np.array(
        microbe_disease_microbe_sorted, dtype=np.int16
    )
    return microbe_disease_microbe_array


if __name__ == "__main__":
    original_result = original_approach(disease_microbe_list, num_microbe)
    generator_result = generator_approach(disease_microbe_list, num_microbe)
