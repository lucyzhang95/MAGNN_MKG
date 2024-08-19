import pandas as pd
from data_preprocess_tools import (
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

# TODO: use NCBI .dmp file to map the out-dated microbial names to taxid
taxdump_path = "taxdump.tar.gz"
mapping_src = load_ncbi_taxdump(taxdump_path)
mapped_microbe_names = map_microbe_name2taxid(
    mapping_src=mapping_src,
    microbe_names=micro_names,
    unmapped_out_path="../data/manual/hmdb_microbe_names_notfound.csv",
)
