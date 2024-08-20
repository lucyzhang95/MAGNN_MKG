import json
import os
import re
import tarfile
from collections import Counter, defaultdict
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


def load_ncbi_taxdump(tar_gz_file_path):
    """
    Map microbial names to taxid using NCBI name.dmp file in taxdump.tar.gz
    fields e.g.,
    ['1115873', 'paraorygmatobothrium taylori cutmore, bennett & cribb, 2009', '', 'authority', '']
    ['1115873', 'paraorygmatobothrium taylori', '', 'scientific name', '']
    output e.g.,
    ('rhinura nr. populonia eo1166', 'rhinura nr. populonia'): {'taxid': 'NCBITaxon:3132808'}
    """
    # open the tar.gz file
    with tarfile.open(tar_gz_file_path, "r:gz") as tar:
        # extract and read the 'names.dmp' file
        with tar.extractfile("names.dmp") as dmp_file:
            dmp_content = dmp_file.read().decode("utf-8")
            lines = dmp_content.splitlines()
            synonym_dict = defaultdict(dict)
            pattern = re.compile(r"\t\|\t|\t\|")

            temp_dict = defaultdict(list)

            for line in lines:
                fields = pattern.split(line.strip())
                if len(fields) >= 5:
                    taxid = fields[0]
                    name_txt = fields[1].lower()
                    name_class = fields[3]
                    if (
                        name_class == "synonym"
                        or name_class == "scientific name"
                        or name_class == "equivalent name"
                    ):
                        temp_dict[taxid].append(name_txt)

            for taxid, synonyms in temp_dict.items():
                synonym_dict[tuple(synonyms)] = {"taxid": f"NCBITaxon:{taxid}"}
            return synonym_dict


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
            if split_char in rec[node][attr]
        ]
    elif attr:
        entity2ct = [
            rec[node][attr]
            for rec in data
            if attr in rec[node] and rec[node][attr] is not None
        ]
    else:
        raise ValueError("Either attribute or split_char must be provided.")

    entity_ct = Counter(entity2ct)
    print(f"{node}_{attr if attr else split_char}: {entity_ct}")


def count_entity4hmdb(
    data: list,
    main_keys: str,
    xrefs_key=None,
):

    main_key_counter = Counter()
    xrefs_key_counter = Counter()
    for rec in data:
        for key in main_keys:
            if key in rec:
                main_key_counter[key] += 1

        if xrefs_key and "xrefs" in rec:
            if xrefs_key in rec["xrefs"]:
                xrefs_key_counter[xrefs_key] += 1

    ct_result = dict(main_key_counter)
    ct_result.update(xrefs_key_counter)
    print(f"Total count of main keys and xrefs keys in records: {ct_result}")
    return ct_result


# TODO: rewrite the function with customizable arg
# def list_filter(a_list, fn):
#   return [item for item in a_list if fn(item)]
#   fn1 = lambda d: d[attr] in inclusion_vals or d[attr] not in exclusion_vals
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


def record_filter(a_list, fn, node=None):
    if node is None:
        filtered_records = [item for item in a_list if fn(item)]
    else:
        filtered_records = [item for item in a_list if fn(item, node)]
    print(f"Total count of filtered records: {len(filtered_records)}")
    return filtered_records


def record_id_filter(a_list, fn, node):
    filtered_ids = [
        item[node]["id"].split(":")[1].strip()
        for item in a_list
        if fn(item, node)
    ]
    print(f"Total count of filtered ids: {len(filtered_ids)}")
    return filtered_ids


def map_disease_id2mondo(
    query,
    scopes: list | str,
    field: list | str,
    unmapped_out_path: str | os.PathLike | None,
) -> dict:
    """
    Use biothings_client to map disease identifiers
    Map (DOID, MeSH, EFO, Orphanet, MedDRA, HP, etc.) to unified MONDO identifier

    :param query: biothings_client query object (a list of objects)
    :param scopes: str or a list of str
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
    get_mondo = bt_disease.querymany(query, scopes=scopes, fields=field)
    query_op = {
        d["query"]: (
            d.get("_id")
            if "notfound" not in d
            else unmapped.append((d["query"], None))
        )
        for d in get_mondo
    }

    mapped = {
        disease_id: mondo for disease_id, mondo in query_op.items() if mondo
    }

    print("count of mapped diseases:", len(mapped))
    print("count of unmapped diseases:", len(unmapped))

    unmapped.sort(key=lambda x: x[0])
    disease_notfound = pd.DataFrame(unmapped, columns=["disease", "mondo"])
    # print("unmapped diseases:", disease_notfound.head())
    disease_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )
    return mapped


# TODO: add unmapped disease names, so that I can embed export path directly in it
def map_disease_name2mondo(
    disease_names: list or str,
    scopes: list or str,
    field: list or str,
    unmapped_out_path: str | os.PathLike | None,
) -> dict:
    """
    Use biothings_client to map disease names to unified MONDO identifier
    Map ("disease_ontology.name" or "disgenet.xrefs.disease_name") to unified MONDO identifier

    :param unmapped_out_path: path to unmapped output file
    :param disease_names: biothings_client query object (a list of disease name strings)
    :param scopes: str or a list of str
    :param field: str or a list of str
    :return: a dictionary of mapped diseases

    query_op exp: {'hyperglycemia': '0002909', 'trichomonas vaginalis infection': None, ...}
    scope and field can be checked via:
    https://docs.mydisease.info/en/latest/doc/data.html#available-fields
    """
    unmapped = []
    disease_names = set(disease_names)
    print("count of unique disease names:", len(disease_names))

    bt_disease = biothings_client.get_client("disease")
    get_mondo = bt_disease.querymany(
        disease_names, scopes=scopes, fields=field
    )
    query_op = {
        d["query"]: (
            d.get("_id")
            if "notfound" not in d
            else unmapped.append(d["query"])
        )
        for d in get_mondo
    }

    # filter out None values
    mapped = {
        disease_name: mondo
        for disease_name, mondo in query_op.items()
        if mondo
    }
    print("count of mapped disease names:", len(mapped))
    print("count of unmapped disease names:", len(unmapped))

    # sort the metabolites by identifier to ensure the order
    unmapped.sort(key=lambda x: x[0])
    disease_names_notfound = pd.DataFrame(unmapped, columns=["disease_name"])
    # print("unmapped disease_name:", disease_names_notfound.head())
    disease_names_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )

    return mapped


def map_metabolite2chebi_cid(
    metabolites: list or str,
    scopes: list or str,
    field: list or str,
    unmapped_out_path: str | os.PathLike | None,
) -> dict:

    unmapped = []
    query_op = {}
    bt_chem = biothings_client.get_client("chem")
    metabolites = set(metabolites)
    print("count of unique metabolites:", len(metabolites))

    # query biothings_client to map metabolites to chebi or pubchem_cid
    get_chebi_cid = bt_chem.querymany(metabolites, scopes=scopes, fields=field)
    for d in get_chebi_cid:
        if "notfound" not in d:
            if d.get("pubchem") and "cid" in d.get("pubchem"):
                query_op[d["query"]] = (
                    f"PUBCHEM.COMPOUND:{d['pubchem']['cid']}"
                )
            elif isinstance(d.get("chebi"), dict):
                query_op[d["query"]] = d["chebi"]["id"]
            else:
                unmapped.append(d["query"])
        else:
            unmapped.append(d["query"])
    mapped = {
        kegg: chebi_cid for kegg, chebi_cid in query_op.items() if chebi_cid
    }
    print("count of mapped unique metabolites:", len(mapped))
    print("count of unmapped unique metabolites:", len(unmapped))

    # sort the metabolites by identifier to ensure the order
    unmapped.sort(key=lambda x: x[0])
    metabolites_notfound = pd.DataFrame(unmapped, columns=["metabolite"])
    # print("unmapped metabolites:", metabolites_notfound.head())
    metabolites_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )

    return mapped


def map_gene2entrez(
    genes: list or str,
    scopes: list or str,
    field: list or str,
    unmapped_out_path: str | os.PathLike | None,
) -> dict:
    unmapped = []
    query_op = {}
    bt_gene = biothings_client.get_client("gene")
    genes = set(genes)
    print("count of unique genes:", len(genes))

    # query biothings_client to map uniportkb to entrezgene(NCBIgene)
    get_entrez = bt_gene.querymany(genes, scopes=scopes, fields=field)
    for d in get_entrez:
        if "notfound" not in d:
            if d.get("entrezgene"):
                query_op[d["query"]] = f"NCBIGene:{d['entrezgene']}"
        else:
            unmapped.append(d["query"])
    mapped = {
        uniprotkb: entrezgene
        for uniprotkb, entrezgene in query_op.items()
        if entrezgene
    }
    print("count of mapped unique genes:", len(mapped))
    print("count of unmapped unique genes:", len(unmapped))

    # sort the genes by identifier to ensure the order
    unmapped.sort(key=lambda x: x[0])
    genes_notfound = pd.DataFrame(unmapped, columns=["uniprotkb"])
    # print("unmapped genes:", genes_notfound.head())
    genes_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )

    return mapped


def map_microbe_name2taxid(mapping_src, microbe_names, unmapped_out_path):
    microbe_names = set(microbe_names)
    print("count of unique microbial names to map:", len(microbe_names))

    mapped = {
        microbe: data["taxid"]
        for microbe in microbe_names
        for synonyms, data in mapping_src.items()
        if microbe in synonyms
    }

    unmapped = [
        microbe
        for microbe in microbe_names
        if not any(microbe in synonyms for synonyms in mapping_src)
    ]

    print("count of mapped unique microbial names:", len(mapped))
    print("count of unmapped unique microbial names:", len(unmapped))

    # sort unmapped microbial names
    unmapped.sort(key=lambda x: x[0])
    names_notfound = pd.DataFrame(unmapped, columns=["microbe_name"])
    # print("unmapped genes:", names_notfound.head())
    names_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )

    return mapped


def get_taxonomy_info(taxids: list, scopes, field, unmapped_out_path):
    unmapped = []
    query_op = {}
    bt_taxon = biothings_client.get_client("taxon")
    taxids = set(taxids)
    print("count of unique taxids:", len(taxids))

    # query biothings_client to get taxonomy information
    get_taxonomy = bt_taxon.querymany(taxids, scopes=scopes, fields=field)
    for d in get_taxonomy:
        if "notfound" not in d:
            query_op[d["query"]] = d
        else:
            unmapped.append(d["query"])

    mapped = {
        taxid: {
            "taxid": taxon["_id"],
            "scientific_name": taxon["scientific_name"],
            "lineage": taxon["lineage"],
            "parent_taxid": taxon["parent_taxid"],
            "rank": taxon["rank"],
        }
        for taxid, taxon in query_op.items()
        if taxon
    }
    print("count of mapped unique taxids:", len(mapped))
    print("count of unmapped unique taxids:", len(unmapped))

    # sort the genes by identifier to ensure the order
    unmapped.sort(key=lambda x: x[0])
    taxid_notfound = pd.DataFrame(unmapped, columns=["uniprotkb"])
    # print("unmapped taxids:", taxon_notfound.head())
    taxid_notfound.to_csv(
        unmapped_out_path, sep="\t", header=True, index=False
    )

    return mapped


# TODO: need to make the function more readable (too many argv now)
# TODO: add more explanation on arg (e.g., attr1: a string represents record key in subj or obj)
def entity_filter_for_magnn(
    data: List[dict],
    node1: str,
    attr1: str,
    val1: Union[str | List[str]],
    node2: str,
    attr2: str,
    attr3: str | None,
) -> List[dict]:
    """
    Final record filter of relational data for MAGNN input
    Final record e.g., [ {'NCBITaxon:9': 'MONDO:0011565'}, {'NCBITaxon:9': 'MESH:D050177'}, ...]

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
        node1_value = rec[node1][attr1]
        node2_value = rec[node2][attr2]

        if attr3 is None:
            # Case when attr3 is None: use node1's attr1 and node2's attr2
            if ":" in node1_value:
                op.append({node1_value: node2_value})
        else:
            # Case when attr3 is provided: use node1's attr3 and node2's attr2
            if node1_value in val1:
                op.append({f"NCBITaxon:{rec[node1][attr3]}": node2_value})
            else:
                # Default case: use node1's attr1 and node2's attr2
                op.append({f"NCBITaxon:{node1_value}": node2_value})

    print("Total records to be exported:", len(op))
    return op


# TODO: assign datatype for data (after merging and before assign index)
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
