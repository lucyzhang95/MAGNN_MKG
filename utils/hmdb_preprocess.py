from data_preprocess_tools import (
    export_data2dat,
    get_taxonomy_info,
    load_data,
    load_ncbi_taxdump,
    map_microbe_name2taxid,
)

# Load HMDB data
data_path = "../data/json/hmdb_microbe_metabolites.json"
data = load_data(data_path)
# microbial names to map (118 unique)
micro_names = [
    taxon.get("scientific_name").strip()
    for rec in data
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

# add microbial taxids and taxon info to original HMDB data
for rec in data:
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

# get the final_op for MAGNN (299 unique)
# e.g., {'NCBITaxon:1227946': 'PUBCHEM.COMPOUND:5280899', 'NCBITaxon:571': 'CHEBI:62064',...}
final_op = {}
for rec in data:
    for microbe in rec["associated_microbes"]:
        if "taxid" in microbe:
            taxid = microbe["taxid"]
            if "pubchem_cid" in rec["xrefs"]:
                pubchem_cid = rec["xrefs"]["pubchem_cid"]
                final_op[f"NCBITaxon:{taxid}"] = pubchem_cid
            else:
                final_op[f"NCBITaxon:{taxid}"] = rec["xrefs"]["chebi"]
# print(final_op)
# print(len(final_op))

export_data2dat(
    in_data=[final_op],
    col1="taxid",
    col2="metabolite_id",
    out_path="../data/MAGNN_data/hmdb_taxid_met.dat",
    database="HMDB",
)
