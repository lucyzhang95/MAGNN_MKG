from biothings.utils.dataload import tabfile_feeder
from data_preprocess_tools import (
    count_entity,
    entity_filter_for_magnn,
    export_data2dat,
    load_data,
    map_disease_name2mondo,
    record_filter_attr1,
)

# Data preprocess for Disbiome Database (10,866)
disbiome_data_path = "../data/json/disbiome_data.json"
disbiome_data = load_data(disbiome_data_path)

# count taxonomic rank records
disbiome_rank_ct = count_entity(
    disbiome_data, node="subject", attr="rank", split_char=None
)

# only want data with taxid and rank == species and strain
# filter record by get parent_taxid if rank is strain, get taxid if rank is species)
disbiome_filtered_by_strain_species = record_filter_attr1(
    disbiome_data,
    node="subject",
    attr="rank",
    include_vals=["strain", "species"],
)

# count disease records before and post filtering by species and strains
# 87 records do not have disease identifiers but only with disease names
disbiome_disease_ct = count_entity(
    disbiome_data, node="object", attr="id", split_char=":"
)
disbiome_filtered_disease_ct = count_entity(
    disbiome_filtered_by_strain_species,
    node="object",
    attr="id",
    split_char=":",
)

# count records with no MONDO ids (4646)
# filtered by records with strains and species and no MONDO
disbiome_filtered_by_md = record_filter_attr1(
    disbiome_filtered_by_strain_species,
    node="object",
    attr=None,
    include_vals=["meddra", "efo", "orphanet", "hp"],
    exclude_vals=["mondo", None],
)
# extract disbiome disease names
disbiome_disease_names = [
    rec["object"]["name"] for rec in disbiome_filtered_by_md
]
print("disease names exclude mondo records:", len(disbiome_disease_names))
print("unique disease names exclude mondo:", len(set(disbiome_disease_names)))

# map disease names to their MONDO identifiers with search for doid names
# 67 dup hits and 145 no hit
# output e.g., {'hyperglycemia': 'MONDO:0002909', ...}
# doidname2mondo = map_disease_name2mondo(
#     disbiome_disease_names,
#     scopes="disease_ontology.name",
#     field="all",
#     unmapped_out_path="../data/manual/disbiome_disease_mondo_notfound.csv",
# )
# print(doidname2mondo)

# load manually mapped disease data
filled_disease_path = (
    "../data/manual/disbiome_disease_notfound_filled_022125.txt"
)
# organize the disease name and MONDO id to a dictionary
# e.g., {'chronic rhinosinusitis': 'MONDO:0006031'}
mapped_disbiome_disease = {}
for line in tabfile_feeder(filled_disease_path, header=0):
    disease_name, mondo, mesh = line[0], line[1], line[2]
    if mondo:
        mapped_disbiome_disease[disease_name] = mondo
    elif mesh:
        mapped_disbiome_disease[disease_name] = mesh
# print(mapped_disbiome_disease)

# add mondo to disbiome_filtered_by_md.object.id
for rec in disbiome_filtered_by_md:
    disease_name = rec["object"]["name"]
    if disease_name in mapped_disbiome_disease:
        rec["object"]["id"] = mapped_disbiome_disease[disease_name]
# print(disbiome_filtered_by_md)

# count the disease identifiers again after manual mapping
# {'MONDO': 4360, 'MESH': 271, 'MedDRA': 14, 'EFO': 1}
disbiome_mapped_disease_ct = count_entity(
    disbiome_filtered_by_md, node="object", attr="id", split_char=":"
)

# extract records with MONDO only (394)
disbiome_filtered_mondo = record_filter_attr1(
    disbiome_filtered_by_strain_species,
    node="object",
    attr=None,
    include_vals=["mondo"],
    exclude_vals=None,
)

# merge the two records together (5,040)
disbiome_final = disbiome_filtered_mondo + disbiome_filtered_by_md
# print(disbiome_final)
print("Total count of merged records with MONDO and MESH", len(disbiome_final))

# final filtered that have records with MONDO identifiers only (4754 = 5040-241-38-6-1)
# final filtered that have records with MONDO and MESH identifiers excluding MedDRA and EFO (5025 = 5040-14-1)
disbiome_final_filtered = [
    rec
    for rec in disbiome_final
    if "MONDO" in rec["object"]["id"] or "MESH" in rec["object"]["id"]
]
print(
    "Total count of records with mondo and mesh", len(disbiome_final_filtered)
)

# count the rank again (5025)
# {'species': 5011, 'strain': 14}
disbiome_final_rank_ct = count_entity(
    disbiome_final_filtered, node="subject", attr="rank", split_char=None
)

if __name__ == "__main__":
    # export microbe and disease from disbiome_final_filtered records (5025)
    # final output e.g., [{'NCBITaxon:59823': 'MONDO:0005010'}, {'NCBITaxon:853': 'MESH:D009765'}, ...]
    disbiome_data4magnn = entity_filter_for_magnn(
        disbiome_final_filtered,
        node1="subject",
        attr1="taxid",
        val1="strain",
        node2="object",
        attr2="id",
        attr3="parent_taxid",
    )
    # print(disbiome_data4magnn)

    # export to .dat (5025->3761; 1291 less due to duplication)
    export_data2dat(
        in_data=disbiome_data4magnn,
        col1="taxid",
        col2="disease_id",
        out_path="../data/MAGNN_data/disbiome_taxid_mondo.dat",
        database="disbiome",
    )
