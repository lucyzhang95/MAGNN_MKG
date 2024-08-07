import json
import os
from collections import Counter
from typing import List, Union

import biothings_client
import pandas as pd


def load_data(data_path: str | os.PathLike) -> list:
    """
    Loads data from JSON files.

    :return: a list of record dictionaries
    """
    with open(data_path, "r") as in_file:
        data = json.load(in_file)
    return data


def count_entity(
    data: list,
    node: str,
    attr: Union[str | None],
    split_char: Union[str, None],
):
    """
     Count the value types associated with the attribute(key) in records
     e.g., count value types of rec["subject"]["rank"] from Disbiome data
     Disbiome:
     "subject":{
       "id":"taxid:216816",
       "taxid":216816,
       "name":"bifidobacterium longum",
       "type":"biolink:Bacterium",
       "scientific_name":"bifidobacterium longum",
       "parent_taxid":1678,
       "lineage":[
          216816,
          1678,
          31953,
          85004,
          1760,
          201174,
          1783272,
          2,
          131567,
          1
       ],
       "rank":"species"
    }

     :param data: json data (list of record dictionaries)
     :param node: str (subject or object)
     :param attr: str or None (dictionary key)
     :param split_char: str or None (delimiter such as ":")
     :return: Counter object (a dictionary of value types)
    """
    if attr and split_char:
        entity2ct = [
            rec[node][attr].split(split_char)[0]
            for rec in data
            if attr in rec[node] and rec[node][attr] is not None
        ]
    elif attr:
        entity2ct = [
            rec[node][attr]
            for rec in data
            if attr in rec[node] and rec[node][attr] is not None
        ]
    else:
        raise ValueError("Either condition or split_char must be provided.")

    entity_ct = Counter(entity2ct)
    print(f"{node}_{attr if attr else split_char}: {entity_ct}")


def record_filter_attr1(
    data: list,
    node: str,
    attr: Union[str, None],
    include_vals: Union[str, List[str]],
    exclude_vals: Union[str, List[str]] = None,
) -> list:
    """
    Filter data records based on one attribute type (key type)
    e.g., only want the records with "rank":"species" and "rank":"strain"

    :param data: json data (list of record dictionaries)
    :param node: str (subject or object)
    :param attr: str or None (dictionary key)
    :param include_vals: str or list of str (attributes or keys want to be included in the record)
    :param exclude_vals: str or list of str (attributes or keys want to be excluded in the record)
    :return: list of record dictionaries
    """
    if isinstance(include_vals, str):
        include_vals = [include_vals]
    if exclude_vals and isinstance(exclude_vals, str):
        exclude_vals = [exclude_vals]

    if attr:
        filtered_records = [
            rec
            for rec in data
            if rec[node].get(attr) in include_vals
            and (not exclude_vals or rec[node].get(attr) not in exclude_vals)
        ]
    else:
        filtered_records = [
            rec
            for rec in data
            if any(val in rec[node] for val in include_vals)
            and (
                not exclude_vals
                or all(ex_val not in rec[node] for ex_val in exclude_vals)
            )
        ]

    print(f"Total count of filtered records: {len(filtered_records)}")
    return filtered_records


def record_filter_attr2(
    data,
    node1: str,
    attr1: str,
    val1: Union[str, List[str]],
    node2: str,
    attr2: Union[str, None],
    val2: Union[str, List[str]],
) -> list:
    """
    Filter data records based on two attribute types (value types)
    e.g., want to get records filtered by both "rank" and "disease_name"

    :param data: json data (list of dictionary records)
    :param node1: str (subject or object)
    :param attr1: str (dictionary key)
    :param val1: str or list of str (values associated with attributes will be included in the record)
    :param node2: str (subject or object)
    :param attr2: str (dictionary key)
    :param val2: str or list of str (values associated with attributes will be included in the record)
    :return: list of record dictionaries
    """
    if isinstance(val1, str):
        val1 = [val1]
        filtered_records = [
            rec
            for rec in data
            if rec[node1].get(attr1) in val1 and val2 in rec[node2]
        ]
    else:
        filtered_records = [
            rec
            for rec in data
            if rec[node1].get(attr1) in val1 and rec[node2].get(attr2) in val2
        ]
    return filtered_records


def map_disease_id2mondo(
    query,
    scope: list | str,
    field: list | str,
    unmapped_out_path: str | os.PathLike | None,
) -> dict:
    """
    Use biothings_client to map disease identifiers
    Map (DOID, MeSH, EFO, Orphanet, MedDRA, HP, etc.) to unified MONDO identifier

    :param query: biothings_client query object (a list of objects)
    :param scope: str or a list of str
    :param field: str or a list of str
    :param unmapped_out_path: path to unmapped output file
    :return: a dictionary of mapped diseases

    query_op exp: {'0000676': '0005083', '0000195': '0011565',...} == {"EFO": "MONDO", ...}
    scope and field can be checked via:
    https://docs.mydisease.info/en/latest/doc/data.html#available-fields
    """
    unmapped = []
    bt_disease = biothings_client.get_client("disease")
    query = set(query)
    print("count of unique disease identifier:", len(query))
    get_mondo = bt_disease.querymany(query, scopes=scope, fields=field)
    query_op = {
        d["query"]: (
            d.get("_id").split(":")[1].strip()
            if "notfound" not in d
            else unmapped.append((d["query"], None))
        )
        for d in get_mondo
    }

    disease_notfound = pd.DataFrame(unmapped, columns=["disease", "mondo"])
    print("unmapped diseases:", disease_notfound.head())
    disease_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )
    return query_op


# TODO: add unmapped disease names, so that I can embed export path directly in it
def map_disease_name2mondo(
    disease_names: list or str, scope: list or str, field: list or str
) -> dict:
    """
    Use biothings_client to map disease names to unified MONDO identifier
    Map ("disease_ontology.name" or "disgenet.xrefs.disease_name") to unified MONDO identifier

    :param disease_names: biothings_client query object (a list of disease name strings)
    :param scope: str or a list of str
    :param field: str or a list of str
    :return: a dictionary of mapped diseases

    query_op exp: {'hyperglycemia': '0002909', 'trichomonas vaginalis infection': None, ...}
    scope and field can be checked via:
    https://docs.mydisease.info/en/latest/doc/data.html#available-fields
    """
    bt_disease = biothings_client.get_client("disease")
    disease_names = set(disease_names)
    get_mondo = bt_disease.querymany(disease_names, scopes=scope, fields=field)
    query_op = {
        d["query"]: d.get("_id").split(":")[1] if "notfound" not in d else None
        for d in get_mondo
    }
    return query_op


# TODO: need to make the function more readable (too many argv now)
def entity_filter_for_magnn(
    data: List[dict],
    node1: str,
    attr1: str,
    val1: Union[str | List[str]],
    node2: str,
    attr2: str,
    attr3: str,
) -> List[dict]:
    """
    Final record filter of relational data for MAGNN input
    Final record exp: [{59823: '0005301'}, {29523: '0004967'}, ...] -> [{taxid: MONDO}, ...]

    :param data: list of record dictionaries
    :param node1: str (subject or object)
    :param attr1: str (dictionary key)
    :param val1: str or list of str (values associated with attributes will be included in the record)
    :param node2: str (subject or object)
    :param attr2: str (dictionary key)
    :param attr3: str (dictionary key)
    :return: list of record dictionaries
    """
    op = []
    if isinstance(val1, str):
        val1 = [val1]

    for rec in data:
        if rec[node1][attr1] in val1:
            op.append(
                {rec[node1][attr3]: rec[node2][attr2].split(":")[1].strip()}
            )
        else:
            op.append(
                {rec[node1][attr1]: rec[node2][attr2].split(":")[1].strip()}
            )
    return op


def export_data2dat(
    in_data: list,
    col1: str,
    col2: str,
    out_path: str | os.PathLike,
    database: str,
):
    """
    Export unique relational pair data to .dat files for MAGNN input
    Final .dat has no header nor index

    :param in_data: list of record dictionaries
    :param col1: str (column name)
    :param col2: str (column name)
    :param out_path: str (path to output file)
    :param database: str for database name (for printing to show final exported unique relational pairs)
    """
    records = [(k, v) for d in in_data for k, v in d.items()]
    print("count of records to be exported:", len(records))
    print("count of records are unique to be exported:", len(set(records)))
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(set(records), columns=[col1, col2])
    df.to_csv(out_path, sep="\t", header=False, index=False)
    print(
        f"* {len(df)} records of {col1}-{col2} from {database} have been exported successfully!"
    )
