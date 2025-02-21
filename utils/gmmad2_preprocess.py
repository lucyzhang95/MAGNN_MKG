from biothings.utils.dataload import tabfile_feeder
from data_preprocess_tools import (
    count_entity,
    entity_filter_for_magnn,
    export_data2dat,
    load_data,
    map_disease_id2mondo,
    map_metabolite2chebi_cid,
    record_filter,
    record_id_filter,
)
from record_filters import (
    is_not_id,
    is_not_pubchem_cid,
    is_organism_and_disease,
    is_small_molecule_and_gene,
    is_small_molecule_and_taxid,
)

# Load full GMMAD2 data (w/ 3 different relational types)
gmmad2_data_path = "../data/json/gmmad2_data.json"
gmmad2_data = load_data(gmmad2_data_path)

# Data preprocess for GMMAD2 microbe-disease data
# filter out records with microbe-disease relationship only (508,141)
gmmad2_md_rec = record_filter(gmmad2_data, is_organism_and_disease, node=None)

# count taxonomic rank types
gmmad2_md_microbe_ct = count_entity(
    gmmad2_md_rec, node="subject", attr="rank", split_char=None
)
# count disease records
gmmad2_md_disease_ct = count_entity(
    gmmad2_md_rec, node="object", attr="id", split_char=":"
)

# extract disease mesh id (a list of mesh ids)
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
    "../data/manual/gmmad2_disease_notfound_filled_022125.txt"
)

# organize the disease mesh id and MONDO id to a dictionary
# e.g., {'D016360': 'MONDO:0024388', 'D000067877': 'MESH:D000067877', ...}
mapped_gmmad2_disease = {}
for line in tabfile_feeder(gmmad2_filled_disease_path, header=1):
    mesh, mondo = line[0], line[1]
    if mondo:
        mapped_gmmad2_disease[mesh] = mondo
    elif mesh:
        mapped_gmmad2_disease[mesh] = f"MESH:{mesh}"
# print(mapped_gmmad2_disease)
print("manually mapped disease count:", len(mapped_gmmad2_disease))

# merge mesh2mondo + mapped mesh (83 unique)
# e.g., {'D037841': 'MESH:D037841', 'D010661': 'MONDO:0009861', ...}
gmmad2_complete_mapped_disease = mesh2mondo | mapped_gmmad2_disease
# print(gmmad2_complete_mapped_disease)
print(
    "Merged mapped unique disease count:",
    len(set(gmmad2_complete_mapped_disease)),
)

# add mapped disease mondo to gmmad2_md_rec.object.id
for rec in gmmad2_md_rec:
    disease_mesh = rec["object"]["mesh"]
    if disease_mesh in gmmad2_complete_mapped_disease:
        rec["object"]["id"] = gmmad2_complete_mapped_disease[disease_mesh]
# print(gmmad2_md_rec)

# count the disease identifiers again after manual mapping
# {'MONDO': 422606, 'MESH': 85535} == total:508,141
gmmad2_mapped_disease_ct = count_entity(
    gmmad2_md_rec, node="object", attr="id", split_char=":"
)

# final record filter for MAGNN (508,141)
gmmad2_md4magnn = entity_filter_for_magnn(
    gmmad2_md_rec,
    node1="subject",
    attr1="taxid",
    val1=["strain", "subspecies", "biotype", "species group"],
    node2="object",
    attr2="id",
    attr3="parent_taxid",
)
# print(gmmad2_md4magnn)

# export the final filtered records to .dat file (508,141-> 502,039; 6102 less)
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
gmmad2_mm_rec = record_filter(
    gmmad2_data, is_small_molecule_and_taxid, node=None
)

# count the metabolite identifier types (265,705 recs do not have identifier)
met_type_ct = count_entity(
    gmmad2_mm_rec, node="object", attr="id", split_char=":"
)

# extract kegg_compound and kegg_glycan from the metabolite records (42,549; unique 110)
met4query = record_id_filter(gmmad2_mm_rec, is_not_pubchem_cid, node="object")

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

# load manually mapped metabolite data (w/ 6 dup hits)
gmmad2_filled_metabolite_path = (
    "../data/manual/gmmad2_metabolite_notfound_022125.txt"
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
print("Merged mapped unique metabolite count:", len(set(met_mapped)))

# add mapped pubchem_sid, pubchem_cid, chebi to gmmad2_mm_rec.object.id
# gmmad2_mmc_rec includes records with and without identifiers (864,357)
for rec in gmmad2_mm_rec:
    met_id = rec["object"]["id"]
    if ":" in met_id and "PUBCHEM.COMPOUND" not in met_id:
        met_kegg = met_id.split(":")[1].strip()
        rec["object"]["id"] = met_mapped.get(met_kegg, met_id)

# filter the records with metabolite identifiers (598,652)
gmmad2_mm_rec_filtered = record_filter(gmmad2_mm_rec, is_not_id, node="object")

# count the metabolite types after mapping (598,652)
# {'PUBCHEM.COMPOUND': 560655, 'CHEBI': 29204, 'PUBCHEM.SUBSTANCE': 8450, 'KEGG.GLYCAN': 343}
met_type_final_ct = count_entity(
    gmmad2_mm_rec_filtered, node="object", attr="id", split_char=":"
)

# TODO: need add the metabolites with no identifiers to GMMAD2 internal identifiers
#  Need to change the original parser to include the internal identifiers
#  metabolites with no chem identifiers (265,705) ~ 1/3 of the total
#  Also need to differentiate KEGG.COMPOUND and KEGG.GLYCAN in original parser

# final record filter for MAGNN input (598,652)
gmmad2_mm4magnn = entity_filter_for_magnn(
    gmmad2_mm_rec_filtered,
    node1="subject",
    attr1="taxid",
    val1=["strain", "subspecies", "biotype", "species group"],
    node2="object",
    attr2="id",
    attr3="parent_taxid",
)
# print(gmmad2_mm4magnn)

# export the final filtered records to .dat file (598,652->598,603 unique; 49 less)
export_data2dat(
    in_data=gmmad2_mm4magnn,
    col1="taxid",
    col2="metabolite",
    out_path="../data/MAGNN_data/gmmad2_taxid_met.dat",
    database="GMMAD2: Microbe-Metabolite",
)


# Data preprocess for GMMAD2 metabolite-gene data
# filter out records with metabolite-gene relationship only (53,278)
gmmad2_mg_rec = record_filter(gmmad2_data, is_small_molecule_and_gene)

# count the gene identifier types
gene_type_ct = count_entity(
    gmmad2_mg_rec, node="object", attr="id", split_char=":"
)

# count the metabolite identifier types
# 1 record does not have metabolite identifier, discard it
mg_met_type_ct = count_entity(
    gmmad2_mg_rec, node="subject", attr="id", split_char=":"
)

if __name__ == "__main__":
    # final record filter for MAGNN input with {pubchem_cid:gene_id} (53,277)
    # e.g., [{'PUBCHEM.COMPOUND:985': 'NCBIGene:2796'}, ...]
    gmmad2_mg4magnn = entity_filter_for_magnn(
        gmmad2_mg_rec,
        node1="subject",
        attr1="id",
        val1=["PUBCHEM.COMPOUND"],
        node2="object",
        attr2="id",
        attr3=None,
    )
    # print(gmmad2_mg4magnn)

    # export the final filtered records to .dat file (53,277->53,277 unique)
    export_data2dat(
        in_data=gmmad2_mg4magnn,
        col1="pubchem_cid",
        col2="gene",
        out_path="../data/MAGNN_data/gmmad2_met_gene.dat",
        database="GMMAD2: Metabolite-Gene",
    )
