from biothings.utils.dataload import tabfile_feeder
from data_preprocess_tools import (
    count_entity,
    entity_filter_for_magnn,
    export_data2dat,
    load_data,
    map_disease_name2mondo,
    record_filter_attr1,
)

# Data preprocess for Disbiome Database
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
# output e.g., {'hyperglycemia': '0002909', 'trichomonas vaginalis infection': None, ...}
doidname2mondo = map_disease_name2mondo(
    disbiome_disease_names, scopes="disease_ontology.name", field="all"
)
# print(doidname2mondo)
# print("Mapping output with None count:", len(doidname2mondo))

# # tested to use different scope such as disgenet.xrefs.disease_name
# # the output is too complicated to process, so ignored
# # output has 111 input dup hits and 113 found no hit for using disgenet2.xrefs.disease_name
# disgenet2mondo = map_disease_name2mondo(get_disease_name, "disgenet.xrefs.disease_name", "all")
# print(len(disgenet2mondo))

# # export all notfound disease names to csv and do manual mapping
# disease_notfound = pd.DataFrame.from_dict(doidname2mondo, orient='index', dtype=str).reset_index()
# print(disease_notfound)
# disease_notfound.columns = ["name", "mondo"]
# disease_notfound.to_csv("../data/manual/disbiome_disease_mondo_notfound.csv")

# load manually mapped disease data
filled_disease_path = (
    "../data/manual/disbiome_disease_notfound_filled_mondo.txt"
)
# organize the disease name and MONDO id to a dictionary e.g., {'chronic rhinosinusitis': '0006031'}
mapped_disbiome_disease = {}
for line in tabfile_feeder(filled_disease_path, header=0):
    filled_disbiome_disease, mondo, mesh = line[0], line[1], line[2]
    if mondo:
        mapped_disbiome_disease[filled_disbiome_disease] = mondo
    elif mesh:
        mapped_disbiome_disease[filled_disbiome_disease] = mesh
print("mapped disease count:", len(mapped_disbiome_disease))

# need to add mondo to the filtered records that do not have mondo
for rec in disbiome_filtered_by_md:
    disease_name = rec["object"]["name"]
    if disease_name in mapped_disbiome_disease:
        if "D" in mapped_disbiome_disease[disease_name]:
            rec["object"][
                "id"
            ] = f"MESH:{mapped_disbiome_disease[disease_name]}"
        else:
            rec["object"][
                "id"
            ] = f"MONDO:{mapped_disbiome_disease[disease_name]}"

# count the disease identifiers again after manual mapping
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

# merge the two records together (5040)
disbiome_final = disbiome_filtered_mondo + disbiome_filtered_by_md
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
# count the rank again
disbiome_final_rank_ct = count_entity(
    disbiome_final_filtered, node="subject", attr="rank", split_char=None
)

# export microbe and disease from disbiome_final_filtered records
# final output e.g., [{59823: '0005301'}, {29523: '0004967'}, ...] -> [{taxid: MONDO}...]
# if rank == "strain", then use parent_taxid instead of taxid
# TODO: leading 0 gets removed when converting identifiers to integer
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
export_data2dat(
    in_data=disbiome_data4magnn,
    col1="taxid",
    col2="mondo",
    out_path="../data/MAGNN_data/disbiome_taxid_mondo.dat",
    database="disbiome",
)
