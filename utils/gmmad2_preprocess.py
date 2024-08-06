from data_preprocess_tools import (
    count_entity,
    load_data,
    map_disease_id2mondo,
    record_filter_attr1,
)

# Data preprocess for GMMAD2 database
gmmad2_data_path = "../data/json/gmmad2_data.json"
gmmad2_data = load_data(gmmad2_data_path)
gmmad2_microbe_ct = count_entity(gmmad2_data, node="subject", attr="rank")

#     gmmad2_disease_data = [rec for rec in gmmad2_data if "OrganismalEntityAsAModelOfDiseaseAssociation" in rec['association']['predicate']]
#     gmmad2_microbes = [item['subject']['name'] for item in gmmad2_disease_data]
#     gmmad2_diseases = [item['object']['name'] for item in gmmad2_disease_data]
#
#     print(f"count of gmmad2 microbes: {len(gmmad2_microbes)}")
#     print(f"count of unique gmmad2 microbes: {len(set(gmmad2_microbes))}")
#
#     # print(f"count of microbial species with taxid: {len(microbe_species)}")
#     # print(f"count of unique microbial species with taxid: {len(set(microbe_species))}")
#     print(f"count of gmmad2 diseases: {len(gmmad2_diseases)}")
#     print(f"count of unique gmmad2 diseases: {len(set(gmmad2_diseases))}")

#     gmmad2_rank2count = [rec['subject'].get('rank') for rec in gmmad2_disease_data]
#     rank_count = Counter(gmmad2_rank2count)
#     print(f"count of taxon rank in GMMAD2: {rank_count}")
#
#     gmmad2df = [
#         {
#             'microbe': record['subject']['name'],
#             'taxid': int(record['subject'].get('taxid')),
#             'disease_name': record['object']['name'],
#             'disease_id': record['object']['mesh'] if "mesh" in record['object'] else None
#         }
#         for record in gmmad2_data
#         if "OrganismalEntityAsAModelOfDiseaseAssociation" in record['association']['predicate']
#         if record['subject'].get('rank') == "species" or "strain"
#     ]
#
# merged_md = disbiome2df + gmmad2df
# print(merged_md)
# print(f"total count of merged md set: {len(merged_md)}")
#
#
# def remove_duplicate_microdis_sets(lst):
#     seen = set()
#     unique_list = []
#     for d in lst:
#         # Create a unique key based on taxid and disease_name
#         md_sets = (d['taxid'], d['disease_name'])
#         if md_sets not in seen:
#             seen.add(md_sets)
#             unique_list.append(d)
#     return unique_list
#
#
# unique_md_sets = remove_duplicate_microdis_sets(merged_md)
# # print(unique_md_sets)
# # print(len(unique_md_sets))
#
# merged_md_df = pd.DataFrame(unique_md_sets)
#
#
# def convert_disease_id(value):
#     if str(value).isdigit():
#         return int(value)
#     else:
#         return str(value)
#
#
# # Apply the custom function to the disease_id column
# merged_md_df['disease_id'] = merged_md_df['disease_id'].apply(convert_disease_id)
# merged_md_df = merged_md_df.dropna()
# merged_md_df = merged_md_df.astype({
#     'microbe': 'str',
#     'taxid': 'Int64',
#     'disease_name': 'str'
# })
#
# merged_md_df.index += 1
# merged_md_df.to_csv("../data/MAGNN_data/merged_microbe_disease.csv")
# merged_md_df[["taxid", "disease_id"]].to_csv('../data/MAGNN_data/merged_microbe_disease.dat', sep='\t', index=False, header=False)
