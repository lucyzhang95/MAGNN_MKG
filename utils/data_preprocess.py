import os
import sys
import pickle
import json
import re

import biothings_client
from biothings.utils.dataload import tabfile_feeder
import pandas as pd
from collections import Counter
from typing import List, Union


def count_entity(data, node, condition=None, split_char=None):
    if condition and split_char:
        entity2ct = [rec[node][condition].split(split_char)[0] for rec in data if condition in rec[node] and rec[node][condition] is not None]
    elif condition:
        entity2ct = [rec[node][condition] for rec in data if condition in rec[node] and rec[node][condition] is not None]
    else:
        raise ValueError("Either condition or split_char must be provided.")

    entity_ct = Counter(entity2ct)
    print(f"{node}_{condition if condition else split_char}: {entity_ct}")


def record_filter_attr1(data, node: str, attr: Union[str, None], include_vals: Union[str, List[str]], exclude_vals: Union[str, List[str]] = None) -> list:
    if isinstance(include_vals, str):
        include_vals = [include_vals]
    if exclude_vals and isinstance(exclude_vals, str):
        exclude_vals = [exclude_vals]

    if attr:
        filtered_records = [
            rec for rec in data
            if rec[node].get(attr) in include_vals and (not exclude_vals or rec[node].get(attr) not in exclude_vals)
        ]
    else:
        filtered_records = [
            rec for rec in data
            if any(val in rec[node] for val in include_vals) and (not exclude_vals or all(ex_val not in rec[node] for ex_val in exclude_vals))
        ]

    print(f"Total count of filtered records: {len(filtered_records)}")
    return filtered_records


def record_filter_attr2(data, node1: str, attr1: str, val1: Union[str, List[str]], node2: str, attr2: Union[str, None],  val2: Union[str, List[str]]) -> list:
    if isinstance(val1, str):
        val1 = [val1]
        filtered_records = [
            rec for rec in data
            if rec[node1].get(attr1) in val1 and val2 in rec[node2]
        ]
    else:
        filtered_records = [
            rec for rec in data
            if rec[node1].get(attr1) in val1 and rec[node2].get(attr2) in val2
            ]
    return filtered_records


def map_disease_id2mondo(query, scope: list | str, field):
    unmapped = []
    bt_disease = biothings_client.get_client("disease")
    query = set(query)
    print("count of unique identifier:", len(query))
    get_mondo = bt_disease.querymany(query, scopes=scope, fields=field)
    # query_op exp: {'0000676': '0005083', '0000195': '0011565'} == {"EFO": "MONDO"}
    query_op = {d["query"]: d.get("_id").split(":")[1].strip() if "notfound" not in d else unmapped.append(d["query"]) for d in get_mondo}
    return query_op


def map_disease_name2mondo(disease_names, scope, field):
    unmapped_diseases = []
    bt_disease = biothings_client.get_client("disease")
    disease_names = set(disease_names)
    get_mondo = bt_disease.querymany(disease_names, scopes=scope, fields=field)
    query_op = {d["query"]: d.get("_id").split(":")[1] if "notfound" not in d else None for d in get_mondo}
    return query_op


def entity_filter_for_magnn(data, node1, attr1, val1, node2, attr2, attr3):
    op = []
    for rec in data:
        if str(rec[node1][attr1]) == str(val1):
            op.append({rec[node1][attr3]: rec[node2][attr2].split(":")[1].strip()})
        else:
            op.append({rec[node1][attr1]: rec[node2][attr2].split(":")[1].strip()})
    return op


def export_data2dat(i_data, col1, col2, o_fname, database):
    records = [(k, v) for d in i_data for k, v in d.items()]
    print("count of records to be exported:", len(records))
    print("count of records are unique to be exported:", len(set(records)))
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(set(records), columns=[col1, col2])
    df.to_csv(o_fname, sep='\t', header=False, index=False)
    print(f"* {len(df)} records of {col1}-{col2} from {database} have been exported successfully!")


if __name__ == "__main__":
    total_md_list = []
    # Data preprocess for Disbiome Database
    disbiome_data_path = "../data/json/disbiome_data.json"
    with open(disbiome_data_path, "r") as input_f:
        # load disbiome data
        disbiome_data = json.load(input_f)

        # count taxonomic rank records
        disbiome_rank_ct = count_entity(disbiome_data, "subject", condition="rank")

        # only want data with taxid and rank == species and strain
        # filter record by get parent_taxid if rank is strain, get taxid if rank is species)
        disbiome_filtered_by_strain_species = record_filter_attr1(disbiome_data, "subject", "rank", include_vals=["strain", "species"])

        # count disease records before and post filtering by species and strains
        # 87 records do not have disease identifiers but only with disease names
        disbiome_disease_ct = count_entity(disbiome_data, "object", condition="id", split_char=":")
        disbiome_filtered_disease_ct = count_entity(disbiome_filtered_by_strain_species, "object", condition="id", split_char=":")

        # count records with no MONDO ids (4646)
        # filtered by records with strains and species and no MONDO
        disbiome_filtered_by_md = record_filter_attr1(disbiome_filtered_by_strain_species, "object", None, ["meddra", "efo", "orphanet", "hp"], exclude_vals=["mondo", None])
        # extract disbiome disease names
        disbiome_disease_names = [rec["object"]["name"] for rec in disbiome_filtered_by_md]
        print("disease names exclude mondo records:", len(disbiome_disease_names))
        print("unique disease names exclude mondo:", len(set(disbiome_disease_names)))

        # map disease names to their MONDO identifiers with search for doid names
        doidname2mondo = map_disease_name2mondo(disbiome_disease_names, "disease_ontology.name", "all")
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
        filled_disease_path = "../data/manual/disbiome_disease_notfound_filled_mondo.txt"
        # organize the disease name and MONDO id to a dictionary e.g., {'chronic rhinosinusitis': '0006031'}
        mapped_disbiome_disease = {}
        for line in tabfile_feeder(filled_disease_path, header=0):
            filled_disbiome_disease, mondo_id = line[0], line[1]
            if mondo_id:
                mapped_disbiome_disease[filled_disbiome_disease] = mondo_id
        # print(complete_disbiome_disease)
        print("mapped disease count:", len(mapped_disbiome_disease))

        # need to add mondo to the filtered records that do not have mondo
        for rec in disbiome_filtered_by_md:
            disease_name = rec["object"]["name"]
            if disease_name in mapped_disbiome_disease:
                rec["object"]["id"] = f"MONDO:{mapped_disbiome_disease[disease_name]}"

        # count the disease identifiers again after manual mapping
        disbiome_mapped_disease_ct = count_entity(disbiome_filtered_by_md, "object", condition="id", split_char=":")

        # extract records with MONDO only (394)
        disbiome_filtered_mondo = record_filter_attr1(disbiome_filtered_by_strain_species, "object", None, include_vals=["mondo"], exclude_vals=None)

        # merge the two records together (5040)
        disbiome_final = disbiome_filtered_mondo + disbiome_filtered_by_md
        # print(len(disbiome_final))

        # final filtered that have records with MONDO identifiers only (4754 = 5040-241-38-6-1)
        disbiome_final_filtered = [rec for rec in disbiome_final if "MONDO" in rec["object"]["id"]]
        print("Total count of records with mondo only", len(disbiome_final_filtered))
        # count the rank again
        disbiome_final_rank_ct = count_entity(disbiome_final_filtered, "subject", "rank")

        # export microbe and disease from disbiome_final_filtered records
        # final output e.g., [{59823: '0005301'}, {29523: '0004967'}, ...] -> [{taxid: MONDO}...]
        # if rank == "strain", then use parent_taxid instead of taxid
        # TODO: leading 0 gets removed when converting to integer
        disbiome_data4magnn = entity_filter_for_magnn(disbiome_final_filtered, "subject", "taxid", "strain", "object", "id", "parent_taxid")
        # print(disbiome_data4magnn)
        export_data2dat(disbiome_data4magnn, "taxid", "mondo", "../data/MAGNN_data/disbiome_taxid_mondo.dat", "disbiome")







# # Data Process for GMMAD2 database
# gmmad2_data_path = "../data/json/gmmad2_data.json"
# with open(gmmad2_data_path, "r") as input_f:
#     gmmad2_data = json.load(input_f)
#
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