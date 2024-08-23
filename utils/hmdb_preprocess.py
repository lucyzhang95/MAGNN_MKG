from biothings.utils.dataload import tabfile_feeder
from data_preprocess_tools import (
    count_entity4hmdb,
    export_data2dat,
    get_taxonomy_info,
    load_data,
    load_ncbi_taxdump,
    map_disease_id2mondo,
    map_microbe_name2taxid,
)

# Load HMDB data
data_path = "../data/json/hmdb_metabolites_full.json"
data = load_data(data_path)

# Data preprocess for microbe-metabolite association data
mm_data = [rec for rec in data if "associated_microbes" in rec]
print("Total records of microbe-metabolite:", len(mm_data))

# microbial names to map (118 unique)
micro_names = [
    taxon.get("scientific_name").strip()
    for rec in mm_data
    if "associated_microbes" in rec
    for taxon in rec.get("associated_microbes")
    if "taxid" not in taxon
]

# Load NCBI taxdump data
# mapped:70, unmapped:48
taxdump_path = "taxdump.tar.gz"
mapping_src = load_ncbi_taxdump(taxdump_path)
mapped_microbe_names = map_microbe_name2taxid(
    mapping_src=mapping_src,
    microbe_names=micro_names,
    unmapped_out_path="../data/manual/hmdb_microbe_names_notfound.csv",
)

# load manually mapped microbial taxids (44)
manual_mapped_names_path = (
    "../data/manual/hmdb_micro_names_notfound_filled.txt"
)
manual_mapped_micro_names = {}
with open(manual_mapped_names_path, "r", encoding="ISO-8859-1") as in_f:
    next(in_f)

    for line in in_f:
        line = line.strip().split("\t")
        if len(line) > 1:
            microbe_name, taxid = line[0], line[1]
            manual_mapped_micro_names[microbe_name] = taxid
# print(manual_mapped_micro_names)
print(
    "Count of mapped unique microbial names:", len(manual_mapped_micro_names)
)

# merge the mapped_microbe_names and manual_mapped_micro_names together (114)
full_microbe_names = mapped_microbe_names.copy() | manual_mapped_micro_names
# print(full_microbe_names)
# print(len(full_microbe_names))

# extract taxids from manual mapped microbial names dictionary (103 unique)
manual_mapped_taxids = [
    v.split(":")[1].strip() for k, v in full_microbe_names.items()
]
# print(manual_mapped_taxids)
get_taxon4manual = get_taxonomy_info(
    manual_mapped_taxids,
    scopes=["taxid"],
    field=["lineage", "scientific_name", "rank", "parent_taxid"],
    unmapped_out_path="../data/manual/hmdb_micro_taxids_notfound.csv",
)
# print(get_taxon4manual)

# merge the manual_mapped_micro_names and get_taxon4manual together (114)
manual_mapped_full_info = {
    k: {
        "taxid": v.split(":")[1].strip(),
        **get_taxon4manual[v.split(":")[1].strip()],
    }
    for k, v in full_microbe_names.items()
    if v.split(":")[1].strip() in get_taxon4manual
}
# print(manual_mapped_full_info)
# print(len(manual_mapped_full_info))

# add microbial taxids and taxon info to mm_data
for rec in mm_data:
    if "associated_microbes" in rec:
        for microbe_d in rec.get("associated_microbes"):
            if "taxid" not in microbe_d:
                if (
                    microbe_d.get("scientific_name").strip()
                    in manual_mapped_full_info
                ):
                    microbe_d.update(
                        manual_mapped_full_info[
                            microbe_d.get("scientific_name").strip()
                        ]
                    )

# get the mm_op for MAGNN (625 unique)
# only use the records with species and strain rank (599)
# e.g., {'NCBITaxon:1227946': 'PUBCHEM.COMPOUND:5280899', 'NCBITaxon:571': 'CHEBI:62064',...}
mm_op = []
for rec in mm_data:
    met_id = rec["xrefs"].get("pubchem_cid", rec["xrefs"].get("chebi"))

    for microbe in rec.get("associated_microbes"):
        taxid_key = (
            f"NCBITaxon:{microbe['taxid']}"
            if microbe.get("rank") == "species"
            else f"NCBITaxon:{microbe.get('parent_taxid')}"
        )
        mm_op.append({taxid_key: met_id})
# print(mm_op)
# print(len(mm_op))

# final export record (599)
export_data2dat(
    in_data=mm_op,
    col1="taxid",
    col2="metabolite_id",
    out_path="../data/MAGNN_data/hmdb_taxid_met.dat",
    database="HMDB",
)


# Data preprocess for metabolite-disease association data (22,600)
disease_rec_ct = [rec for rec in data if "associated_diseases" in rec]
print(
    "Total count for records including associated_diseases:",
    len(disease_rec_ct),
)

# count pubchem_cid with associated diseases  (13044; 13037 unique)
pubchem_cid_ct = count_entity4hmdb(
    disease_rec_ct, main_keys="_id", xrefs_key="pubchem_cid"
)
# count chebi with associated diseases (1989)
chebi_ct = count_entity4hmdb(
    disease_rec_ct, main_keys="_id", xrefs_key="chebi"
)
# count inchikey with associated diseases (22,600)
inchikey_ct = count_entity4hmdb(
    disease_rec_ct, main_keys="_id", xrefs_key="inchikey"
)

"""
# extract disease names for mondo or mesh query (27670, 657 unique)
disease_names4query = [
    disease["name"].strip()
    for rec in data
    if "associated_diseases" in rec
    for disease in rec.get("associated_diseases")
]
# print(set(disease_names))
print("Total count of disease names:", len(disease_names4query))

# extract omim ids for mondo query (391 unique)
omim4query = [
    disease["omim"].split(":")[1].strip()
    for rec in data
    if "associated_diseases" in rec
    for disease in rec.get("associated_diseases")
    if "omim" in disease
]
print(len(omim4query))

# map disease names to mondo
# 114 dup hits, 340 no hits
name2mondo = map_disease_name2mondo(
    disease_names4query,
    scopes=["disease_ontology.name"],
    field=["all"],
    unmapped_out_path="../data/manual/hmdb_disease_names_notfound.csv",
)
print(name2mondo)

# map omim ids to mondo (373 unique)
# 18 no hits
omim2mondo = map_disease_id2mondo(
    omim4query,
    scopes=["mondo.xrefs.omim"],
    field=["mondo"],
    unmapped_out_path="../data/manual/hmdb_disease_omim_notfound.csv",
)
print(omim2mondo)
"""

# get omim or disease names if no omim (646)
disease4query = [
    (
        disease.get("omim").split(":")[1].strip()
        if "omim" in disease
        else disease["name"]
    )
    for rec in data
    if "associated_diseases" in rec
    for disease in rec.get("associated_diseases")
]
# print(set(disease4query))
print(
    "Total count of unique omim and disease names for query:",
    len(set(disease4query)),
)

# map disease names and omim ids to mondo (467)
# 50 dup hits, 179 no hits
disease2mondo = map_disease_id2mondo(
    disease4query,
    scopes=["mondo.xrefs.omim", "disease_ontology.name"],
    field=["mondo"],
    unmapped_out_path="../data/manual/hmdb_disease_notfound.csv",
)

# load manually mapped disease data
manual_disease_path = "../data/manual/hmdb_disease_notfound_filled.txt"
manual_mapped_diseases = {}
# organize the disease name and mapped ids to a dictionary (180)
# 1 no mapping (oxidative stress)
# MONDO:161, MESH:16, UMLS:3
# e.g., { 'tuberculous meningitis': 'MONDO:0006042', 'nonketotic hyperglycinemia': 'MESH:D020158', ...}
for line in tabfile_feeder(manual_disease_path, header=1):
    # Organize disease names and mapped IDs into a dictionary
    disease_name, mondo, mesh, doid, umls = line[:5]

    # Prioritize mapping in the order: MONDO, MESH, DOID, UMLS
    manual_mapped_diseases[disease_name] = next(
        (disease_id for disease_id in [mondo, mesh, doid, umls] if disease_id),
        None,
    )

# remove None values
manual_mapped_diseases = {k: v for k, v in manual_mapped_diseases.items() if v}
# print(manual_mapped_diseases)
# print(len(manual_mapped_diseases))
# print(len(set(manual_mapped_diseases)))

# merge all mapped diseases (645)
# MONDO:626, MESH:16, UMLS:3
full_mapped_diseases = disease2mondo.copy() | manual_mapped_diseases
# print(full_mapped_diseases)
print("Total count of mapped diseases:", len(full_mapped_diseases))

# add mapped diseases to disease_ct_rec.associated_diseases.id
for rec in disease_rec_ct:
    associated_diseases = rec.get("associated_diseases")
    for disease in associated_diseases:
        disease_name = disease.get("name")
        omim = disease.get("omim")
        omim_id = omim.split(":")[1].strip() if omim else None

        mapped_id = full_mapped_diseases.get(
            disease_name
        ) or full_mapped_diseases.get(omim_id)
        if mapped_id:
            disease["id"] = mapped_id

# count the disease identifiers again after manual mapping (27652)
# MONDO: 27,535, MESH: 100, UMLS: 17 = 27,652
mondo2ct = [
    disease["id"].split(":")[0]
    for rec in disease_rec_ct
    for disease in rec.get("associated_diseases")
    if "id" in disease
]
# print(mondo2ct)
# print(len(mondo2ct))

# get the record dictionary for MAGNN (27,652)
metdisease_op = [
    {
        rec["xrefs"].get("pubchem_cid")
        or rec["xrefs"].get("chebi")
        or f"HMDB:{rec['_id']}": disease["id"]
    }
    for rec in disease_rec_ct
    for disease in rec.get("associated_diseases")
    if disease.get("id")
]
# print(metdisease_op)
# print(len(metdisease_op))

# export the unique metabolite-disease associations (27,546)
# HMDB: 9548, PUBCHEM.COMPOUND: 17,887, CHEBI:111 = 27,546
export_data2dat(
    in_data=metdisease_op,
    col1="metabolite_id",
    col2="disease_id",
    out_path="../data/MAGNN_data/hmdb_met_disease.dat",
    database="HMDB",
)
