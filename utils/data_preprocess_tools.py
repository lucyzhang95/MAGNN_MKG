import json
from collections import Counter
from typing import List, Union

import biothings_client
import pandas as pd


def load_data(data_path):
    with open(data_path, "r") as in_file:
        data = json.load(in_file)
    return data


def count_entity(data, node, attr=None, split_char=None):
    """

    :param data: path to data file
    :param node:
    :param attr:
    :param split_char:
    :return:
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
    data,
    node: str,
    attr: Union[str, None],
    include_vals: Union[str, List[str]],
    exclude_vals: Union[str, List[str]] = None,
) -> list:
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


def map_disease_id2mondo(query, scope: list | str, field):
    unmapped = []
    bt_disease = biothings_client.get_client("disease")
    query = set(query)
    print("count of unique identifier:", len(query))
    get_mondo = bt_disease.querymany(query, scopes=scope, fields=field)
    # query_op exp: {'0000676': '0005083', '0000195': '0011565'} == {"EFO": "MONDO"}
    query_op = {
        d["query"]: (
            d.get("_id").split(":")[1].strip()
            if "notfound" not in d
            else unmapped.append(d["query"])
        )
        for d in get_mondo
    }
    return query_op


def map_disease_name2mondo(disease_names, scope, field):
    bt_disease = biothings_client.get_client("disease")
    disease_names = set(disease_names)
    get_mondo = bt_disease.querymany(disease_names, scopes=scope, fields=field)
    query_op = {
        d["query"]: d.get("_id").split(":")[1] if "notfound" not in d else None
        for d in get_mondo
    }
    return query_op


def entity_filter_for_magnn(data, node1, attr1, val1, node2, attr2, attr3):
    op = []
    for rec in data:
        if str(rec[node1][attr1]) == str(val1):
            op.append(
                {rec[node1][attr3]: rec[node2][attr2].split(":")[1].strip()}
            )
        else:
            op.append(
                {rec[node1][attr1]: rec[node2][attr2].split(":")[1].strip()}
            )
    return op


def export_data2dat(i_data, col1, col2, o_fname, database):
    records = [(k, v) for d in i_data for k, v in d.items()]
    print("count of records to be exported:", len(records))
    print("count of records are unique to be exported:", len(set(records)))
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(set(records), columns=[col1, col2])
    df.to_csv(o_fname, sep="\t", header=False, index=False)
    print(
        f"* {len(df)} records of {col1}-{col2} from {database} have been exported successfully!"
    )
