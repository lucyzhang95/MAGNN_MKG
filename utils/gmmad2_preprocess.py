from biothings.utils.dataload import tabfile_feeder
from data_preprocess_tools import (
    count_entity,
    entity_filter_for_magnn,
    export_data2dat,
    load_data,
    map_disease_id2mondo,
    map_metabolite2chebi_cid,
    record_filter,
    record_filter_attr1,
    record_id_filter,
)
from record_filters import (
    is_not_id,
    is_not_pubchem_cid,
    is_small_molecule_and_gene,
    is_small_molecule_and_taxid,
)

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
    scopes=["ctd.mesh"],
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

# add mondo to the filtered records without mondo
for rec in gmmad2_md_rec:
    disease_mesh = rec["object"]["mesh"]
    if disease_mesh in gmmad2_complete_mapped_disease:
        if "D" not in gmmad2_complete_mapped_disease[disease_mesh]:
            rec["object"][
                "id"
            ] = f"MONDO:{gmmad2_complete_mapped_disease[disease_mesh]}"
            rec["object"]["mondo"] = gmmad2_complete_mapped_disease[
                disease_mesh
            ]
# print(gmmad2_md_rec)

# count the disease identifiers again after manual mapping (total:508,141)
gmmad2_mapped_disease_ct = count_entity(
    gmmad2_md_rec, node="object", attr="id", split_char=":"
)

# final record filter for MAGNN input with {taxid:mondo} only (508,141)
gmmad2_md4magnn = entity_filter_for_magnn(
    gmmad2_md_rec,
    node1="subject",
    attr1="taxid",
    val1=["strain", "subspecies", "biotype", "species group"],
    node2="object",
    attr2="id",
    attr3="parent_taxid",
)
# export the final filtered records to .dat file (495,936; 12,205 less)
export_data2dat(
    in_data=gmmad2_md4magnn,
    col1="taxid",
    col2="mondo",
    out_path="../data/MAGNN_data/gmmad2_taxid_mondo.dat",
    database="GMMAD2: Microbe-Disease",
)

# TODO: Question-How to just run the blocks of codes below instead of the entire file?
#  Jupyter notebook or put everything into functions or select the blocks of codes to run (check)
#  debugging mode (line or block)
# Data preprocess for GMMAD2 microbe-metabolite data
# filter out records with microbe-metabolite relationship only (864,357)
gmmad2_mm_rec = record_filter(gmmad2_data, is_small_molecule_and_taxid)

# count the metabolite identifier types (265,705 recs do not have identifier)
met_type_ct = count_entity(
    gmmad2_mm_rec, node="object", attr="id", split_char=":"
)

# extract kegg_compound and kegg_glycan from the metabolite records (42,549; unique 110)
met4query = record_id_filter(gmmad2_mm_rec, is_not_pubchem_cid)

# map kegg_compound and kegg_glycan to pubchem_cid or chebi (6 dup hits and 63 no hit)
# output dict values: pubchem.cid or chebi.id
# output exp: {'C00805': 'CHEBI:16914', 'C04105': 'PUBCHEM.COMPOUND:542', ...}
met2chebi_cid = map_metabolite2chebi_cid(
    met4query,
    scopes=[
        "chebi.xrefs.kegg_compound",
        "chebi.xrefs.kegg_glycan",
    ],
    field=["chebi.id", "pubchem.cid"],
    unmapped_out_path="../data/manual/gmmad2_met_unmapped.csv",
)

# load manually mapped metabolite data
gmmad2_filled_metabolite_path = (
    "../data/manual/gmmad2_metabolite_notfound_filled.txt"
)

# organize the kegg metabolite id and identifiers to a dictionary
# dictionary value types: CHEBI, PUBCHEM.SUBSTANCE, PUBCHEM.COMPOUND, KEGG.COMPOUND, KEGG.GLYCAN
# e.g., {'C00125': 'CHEBI:15991', 'C04688': 'PUBCHEM.SUBSTANCE:7269', ...}
met_manual = {}
for line in tabfile_feeder(gmmad2_filled_metabolite_path, header=1):
    kegg, pubchem, chebi, ccsd = line[0], line[1], line[2], line[3]
    if chebi:
        met_manual[kegg] = chebi
    elif pubchem:
        met_manual[kegg] = pubchem
    else:
        met_manual[kegg] = (
            f"KEGG.COMPOUND:{kegg}" if "C" in kegg else f"KEGG.GLYCAN:{kegg}"
        )
# print(met_manual)

# merge met2chebi_cid and manual mapped metabolites (110 unique)
met_mapped = met2chebi_cid | met_manual

# add mapped pubchem_sid, pubchem_cid, chebi to gmmad2_mm_rec.object.id
# gmmad2_mmc_rec includes records with and without identifiers (864,357)
for rec in gmmad2_mm_rec:
    met_id = rec["object"]["id"]
    if ":" in met_id and "PUBCHEM.COMPOUND" not in met_id:
        met_kegg = met_id.split(":")[1].strip()
        rec["object"]["id"] = met_mapped.get(met_kegg, met_id)

# filter the records with metabolite identifiers (598,652)
gmmad2_mm_rec_filtered = record_filter(gmmad2_mm_rec, is_not_id)

# count the metabolite types after mapping
met_type_final_ct = count_entity(
    gmmad2_mm_rec_filtered, node="object", attr="id", split_char=":"
)

# TODO: need add the metabolites with no identifiers to GMMAD2 internal identifiers
#  Need to change the original parser to include the internal identifiers
#  metabolites with no chem identifiers (265,705) ~ 1/3 of the total
#  Also need to differentiate KEGG.COMPOUND and KEGG.GLYCAN in original parser

# final record filter for MAGNN input with {taxid:met_id} (598,652)
gmmad2_mm4magnn = entity_filter_for_magnn(
    gmmad2_mm_rec_filtered,
    node1="subject",
    attr1="taxid",
    val1=["strain", "subspecies", "biotype", "species group"],
    node2="object",
    attr2="id",
    attr3="parent_taxid",
)

# export the final filtered records to .dat file (598,104 unique; 548 less)
export_data2dat(
    in_data=gmmad2_mm4magnn,
    col1="taxid",
    col2="metabolite",
    out_path="../data/MAGNN_data/gmmad2_taxid_met.dat",
    database="GMMAD2: Microbe-Metabolite",
)
