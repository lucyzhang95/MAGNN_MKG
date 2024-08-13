from biothings.utils.dataload import tabfile_feeder
from data_preprocess_tools import (
    count_entity,
    load_data,
    map_disease_id2mondo,
    record_filter_attr1,
    entity_filter_for_magnn,
    export_data2dat,
    record_filter,
    map_metabolite2inchikey
)
from record_filters import is_small_molecule_and_taxid, is_small_molecule_and_gene

# Data preprocess for GMMAD2 microbe-disease data
gmmad2_data_path = "../data/json/gmmad2_data.json"
gmmad2_data = load_data(gmmad2_data_path)

# filter out records with microbe-disease relationship only
gmmad2_md_rec = record_filter_attr1(
    data=gmmad2_data,
    node="association",
    attr="predicate",
    include_vals="OrganismalEntityAsAModelOfDiseaseAssociation",
    exclude_vals=None,
)

# count rank types
# except species rank, the other types need to use parent_taxid corresponding to species
gmmad2_md_microbe_ct = count_entity(
    gmmad2_md_rec, node="subject", attr="rank", split_char=None
)
# count disease records
gmmad2_md_disease_ct = count_entity(
    gmmad2_md_rec, node="object", attr="id", split_char=":"
)

# extract disease mesh id
gmmad2_md_mesh = [
    rec["object"]["mesh"] for rec in gmmad2_md_rec if "mesh" in rec["object"]
]
# convert mesh identifier to MONDO
# mesh2mondo exp: {'D000086382': '0100096', ...} == {"mesh": "mondo"}
mesh2mondo = map_disease_id2mondo(
    query=gmmad2_md_mesh,
    scope=["ctd.mesh"],
    field="mondo",
    unmapped_out_path="../data/manual/gmmad2_md_unmapped.csv",
)
# print(mesh2mondo)

# load manually mapped disease data
gmmad2_filled_disease_path = (
    "../data/manual/gmmad2_disease_notfound_filled_mondo.txt"
)

# organize the disease mesh id and MONDO id to a dictionary
# e.g., {'D016360': '0024388', ...} == {"mesh": "mondo"}
mapped_gmmad2_disease = {}
for line in tabfile_feeder(gmmad2_filled_disease_path, header=1):
    mesh, mondo = line[0], line[1]
    if mondo:
        mapped_gmmad2_disease[mesh] = mondo
    elif mesh:
        mapped_gmmad2_disease[mesh] = mesh
# print(mapped_gmmad2_disease)
print("manually mapped disease count:", len(mapped_gmmad2_disease))

# merge mesh2mondo + mapped mesh
# merged output exp:
# {'D037841': 'D037841', 'D010661': '0009861', ...} == {"mesh": "mesh", "mesh": "mondo", ...}
gmmad2_complete_mapped_disease = mesh2mondo | mapped_gmmad2_disease
# print(gmmad2_complete_mapped_disease)
print(
    "Merged mapped unique disease count:",
    len(set(gmmad2_complete_mapped_disease)),
)

# TODO: how to make the code below cleaner?
# need to add mondo to the filtered records that do not have mondo
for rec in gmmad2_md_rec:
    disease_mesh = rec["object"]["mesh"]
    if disease_mesh in gmmad2_complete_mapped_disease:
        if "D" not in gmmad2_complete_mapped_disease[disease_mesh]:
            rec["object"]["id"] = f"MONDO:{gmmad2_complete_mapped_disease[disease_mesh]}"
            rec["object"]["mondo"] = gmmad2_complete_mapped_disease[disease_mesh]
# print(gmmad2_md_rec)

# count the disease identifiers again after manual mapping (total:508,141)
gmmad2_mapped_disease_ct = count_entity(
    gmmad2_md_rec, node="object", attr="id", split_char=":"
)

# final record filter for MAGNN input with {taxid:mondo} only (508,141)
gmmad2_data4magnn = entity_filter_for_magnn(gmmad2_md_rec,
                                            node1="subject",
                                            attr1="taxid",
                                            val1=["strain", "subspecies", "biotype", "species group"],
                                            node2="object",
                                            attr2="id",
                                            attr3="parent_taxid")
# export the final filtered records to .dat file (495,936; 12,205 less)
export_data2dat(
    in_data=gmmad2_data4magnn,
    col1="taxid",
    col2="mondo",
    out_path="../data/MAGNN_data/gmmad2_md_taxid_mondo.dat",
    database="GMMAD2:Microbe-Disease",
)


# Data preprocess for GMMAD2 microbe-metabolite data
# filter out records with microbe-metabolite relationship only (864,357)
gmmad2_mm_rec = record_filter(gmmad2_data, is_small_molecule_and_taxid)

# count the metabolite identifier types (265,705 recs do not have pubchem_cid or kegg.compound)
met_type_ct = count_entity(gmmad2_mm_rec, node="object", attr="id", split_char=":")

# extract metabolites with no chem identifiers (265,705)
met4query = [rec["object"].get("chemical_formula") for rec in gmmad2_mm_rec if ":" not in rec["object"]["id"]]
print(len(met4query))
# 193 dup hits, 307 no hit, 14 with 1 hit = unique metabolites: 514
# TODO: not reliable to query the chemical formula, so need to leave it as molecular formula or name or discard
# met2inchikey = map_metabolite2inchikey(met4query, scope=["pubchem.molecular_formula"], field="_id", unmapped_out_path="../data/manual/gmmad2_met_unmapped.csv")
# TODO: need to map pubchem_cid and kegg.compound to inchikey

