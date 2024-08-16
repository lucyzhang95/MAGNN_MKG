from data_preprocess_tools import load_data
import pandas as pd

data_path = "../data/json/hmdb_microbe_metabolites.json"
data = load_data(data_path)
micro_names = [
    taxon.get("scientific_name")
    for rec in data
    if "associated_microbes" in rec
    for taxon in rec.get("associated_microbes")
    if "taxid" not in taxon
]

micro_names.sort(key=lambda x: x[0])
micro_names_df = pd.DataFrame(micro_names, columns=["microbe_name"])
name_out = "../data/manual/hmdb_microbe_names_notfound.csv"
micro_names_df.to_csv(name_out, sep="\t", header=True, index=False)

# TODO: use NCBI .dmp file to map the out-dated microbial names to taxid
