import csv
import glob
import json
import os
import pathlib
import re

import uuid
from collections import defaultdict
from collections.abc import Iterator

import biothings_client
import requests


def get_taxon_info(_ids: Iterator[list]) -> Iterator[dict]:
    """retrieve ncbi taxonomy information from biothings_client
    taxonomy information includes "scientific_name", "parent_taxid", "lineage", and "rank"

    :param _ids: a set of ncbi taxon ids obtained from Disbiome Experiment database "organism_ncbi_id"
    :return: a dictionary with query _ids (taxids) as the keys and taxonomy fields with the query _id as values
    """
    _ids = set(_ids)
    t = biothings_client.get_client("taxon")
    taxon_info = t.gettaxa(_ids, fields=["scientific_name", "parent_taxid", "lineage", "rank"])
    taxon_info = {t["query"]: t for t in taxon_info if "notfound" not in t.keys()}
    yield taxon_info


def get_mondo_doid_id(meddra_ids: Iterator[list]) -> Iterator[dict]:
    """retrieve mapped mondo or disease ontology id with meddra_id

    :param meddra_ids: a set of meddra_ids (diseases) obtained from Disbiome Experiment database "meddra_id"
    :return: a dictionary with mondo_id or disease ontology id as the keys and meddra_ids as the values
    """
    meddra_ids = set(meddra_ids)
    d = biothings_client.get_client("disease")
    query_op = d.querymany(
        meddra_ids,
        scopes=["mondo.xrefs.meddra"],
        fields=["mondo.xrefs.meddra"],
    )
    query_op = {d["query"]: d.get("_id") for d in query_op if "notfound" not in query_op}
    yield query_op


def get_meddra_mapping(files: str | pathlib.Path, meddra_ids: list) -> Iterator[list]:
    """map the disbiome meddra_ids to doid, efo, hp, and orphanet ids

    :param files: /Disbiome/mappings/*.tsv
    :param meddra_ids: a set of meddra_ids retrieved from disbiome experiment database
    :return: a list of mapped meddra_id:doid, efo, hp, and orphanet id pairs
             e.g. [{'10002026': 'Orphanet:803'}, {'10016207': 'Orphanet:342'}, ...]
    """
    mapped_l = []
    for file in files:
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for data in reader:
                meddra_id = data[0].split(":")[1]
                if meddra_id in meddra_ids:
                    mapped_l.append({meddra_id: data[2]})
    yield mapped_l


def get_mapping_files() -> Iterator[str | pathlib.Path]:
    """fetch the meddra_id mappings files downloaded from
    https://www.ebi.ac.uk/spot/oxo/datasources/MedDRA
    include mappings between meddra_ids to doid, efo, hp, and orphanet
    files: /Disbiome/mappings/*.tsv

    :return: a list of file paths
    """
    path = os.path.join(pathlib.Path.cwd(), "mappings/")
    files = glob.glob(f"{path}*.tsv")
    assert files
    yield files


def meddra_mapping_dict(meddra_id: list | set[str]) -> Iterator[defaultdict]:
    """convert the mapping list to a dict with meddra_id as key and other id list as value

    :param meddra_id: a set of meddra_ids retrieved from disbiome experiment database
    :return: a defaultdict e.g. defaultdict(<class 'list'>, {'10002026': ['Orphanet:803', 'EFO:0000253'], ...}
    """
    mapping_dict = defaultdict(list)
    for files in get_mapping_files():
        obj = get_meddra_mapping(files, meddra_id)
        for mapping_l in obj:
            for mapping in mapping_l:
                for k, v in mapping.items():
                    mapping_dict[k].append(v)
    yield mapping_dict


def update_subject_node(subject_node: dict, bt_taxon: dict):
    """update the subject_node dictionary with the info from biothings_client taxon info
    old keys: "organism_ncbi_id", "organism_name"
    new keys: "scientific_name", "parent_taxid", "lineage", "rank"

    :param subject_node: a dictionary with ncbi organism taxon info from Disbiome Experiment database
    :param bt_taxon: a dictionary with ncbi organism taxon info retrieved from biothings_client
    :return: an updated dictionary with info from both Disbiome and biothings_client
    """
    if subject_node["id"].split(":")[1] in bt_taxon:
        taxon_info = bt_taxon[subject_node["id"].split(":")[1]]
        new_taxon_keys = ["scientific_name", "parent_taxid", "lineage", "rank"]
        for key in new_taxon_keys:
            subject_node[key] = taxon_info.get(key)


def get_publication() -> Iterator[dict]:
    """retrieve publication relevant information regarding the Disbiome experiment data
    from https://disbiome.ugent.be:8080/publication

    :return: a dictionary with info retrieved from Disbiome publication database
    :return keys: "publication_id", "title", "pubmed_url", and "pmcid" or "pmid"
    """
    pub_url = "https://disbiome.ugent.be:8080/publication"
    pub_resp = requests.get(pub_url)
    pub_content = json.loads(pub_resp.content)
    pub_all = {}
    for pub in pub_content:
        pubmed_url = pub.get("pubmed_url")
        doi = pub.get("doi")

        pub_dict = {
            "publication_id": pub["publication_id"],
            "title": pub["title"],
        }
        pub_all.update({pub_dict["publication_id"]: pub_dict})

        if pubmed_url:
            pub_dict = {
                "publication_id": pub["publication_id"],
                "title": pub["title"],
                "type": "biolink:Publication",
            }
            pub_all.update({pub_dict["publication_id"]: pub_dict})
            # some pubmed_url have pmid and some have pmcid in it, some don't follow those rules at all
            # "pubmed_url": "https://www.ncbi.nlm.nih.gov/pubmed/25446201"
            # "pubmed_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4646740/"
            pmid_match = re.match(r".*?\/(\d+)", pubmed_url)
            pmcid_match = re.match(r".*?(PMC)(\d+)", pubmed_url)
            if pmcid_match:
                pub_dict["pmcid"] = pmcid_match.groups()[1]
                pub_all.update({pub_dict["publication_id"]: pub_dict})
            elif pmid_match:
                pub_dict["pmid"] = pmid_match.groups()[0]
                pub_all.update({pub_dict["publication_id"]: pub_dict})
            else:
                pub_dict["pubmed_url"] = pub["pubmed_url"]
                pub_all.update({pub_dict["publication_id"]: pub_dict})
        if doi:
            pub_dict["doi"] = pub["doi"]
            pub_all.update({pub_dict["publication_id"]: pub_dict})
    yield pub_all


def get_association(content_dict, keys) -> dict:
    """create association dictionary with different key contents
    some json objects from Disbiome do not have "sample_name", "host_type", or "control_name"

    :param content_dict: a json object retrieved from https://disbiome.ugent.be:8080/experiment
    :param keys: a list of "sample_name", "host_type", "control_name" will be used as dictionary keys
    :return: an association dictionary with different key contents
    """
    association = {
        "predicate": "OrganismalEntityAsAModelOfDiseaseAssociation",
        "qualifier": content_dict["qualitative_outcome"].lower(),
    }

    for key in keys:
        if key in content_dict:
            if content_dict[key]:
                if key == "method_name":
                    association[key] = content_dict[key]
                else:
                    association[key] = content_dict[key].lower()

    if "reduced" in association.get("qualifier", ""):
        association["qualifier"] = association["qualifier"].replace("reduced", "decreased")
    if "elevated" in association.get("qualifier", ""):
        association["qualifier"] = association["qualifier"].replace("elevated", "increased")

    return association


def load_disbiome_data() -> Iterator[dict]:
    """load data from Disbiome Experiment database
    10837 records originally in Disbiome Experiment data
    10742 records have ncbi_taxid and 95 records do not
    8420 records without duplicated out_put dictionary _id
    ncbi_taxid: 1535 without duplicates
    meddra_id: 322 without duplicates, 46 meddra_id are mapped to MONDO (276 has no mapping)

    :param data_path: **arg for biothings_cli to work
    :return: an iterator of dictionary composed of subject_node, object_node, association_node, and publication_node
    """
    # The experimental and publication data from Disbiome is on a static webpage
    exp_url = "https://disbiome.ugent.be:8080/experiment"

    exp_resp = requests.get(exp_url)
    exp_content = json.loads(exp_resp.content)

    # by_taxon is a generator object, so can be only used once
    bt_taxon = get_taxon_info(
        js["organism_ncbi_id"] for js in exp_content if js["organism_ncbi_id"]
    )
    # convert the generator object to a list
    taxons = list(bt_taxon)

    bt_disease = get_mondo_doid_id(js["meddra_id"] for js in exp_content if js["meddra_id"])

    # remove the None mondo dictionary values
    bt_disease = {
        meddra: mondo
        for disease_obj in bt_disease
        for meddra, mondo in disease_obj.items()
        if mondo
    }

    meddra_ids = set([str(js["meddra_id"]) for js in exp_content if js["meddra_id"]])
    meddra_mappings = {k: v for d in meddra_mapping_dict(meddra_ids) for k, v in d.items()}

    # get publication information
    pub_data = get_publication()
    publications = list(pub_data)

    for js in exp_content:
        if js["organism_ncbi_id"]:
            subject_node = {
                "id": f"taxid:{str(js['organism_ncbi_id'])}",
                "taxid": js["organism_ncbi_id"],
                "name": js["organism_name"].lower(),
                "type": "biolink:OrganismalEntity",
            }
            for taxon in taxons:
                update_subject_node(subject_node, taxon)

            if "lineage" in subject_node:
                if 2 in subject_node["lineage"]:
                    subject_node["type"] = "biolink:Bacterium"
                elif 10239 in subject_node["lineage"]:
                    subject_node["type"] = "biolink:Virus"
                elif 4751 in subject_node["lineage"]:
                    subject_node["type"] = "biolink:Fungus"
                elif 2157 in subject_node["lineage"]:
                    subject_node["type"] = "biolink:Archaea"
                else:
                    subject_node["type"] = "biolink:OrganismalEntity"
        else:
            subject_node = {
                "name": js["organism_name"].lower(),
                "type": "biolink:OrganismalEntity",
            }

        object_node = {
            "id": None,
            "name": js["disease_name"].lower(),
            "type": "biolink:Disease",
        }

        if js["meddra_id"]:
            object_node.update(
                {
                    "meddra": str(js["meddra_id"]),
                    "meddra_level": js["meddra_level"],
                }
            )

            if object_node["meddra"] in bt_disease:
                mondo_id = bt_disease[object_node["meddra"]]
                object_node["id"] = mondo_id
                object_node["mondo"] = mondo_id.split(":")[1]
            elif object_node["meddra"] in meddra_mappings:
                # meddra_mappings e.g.{'10002026': ['Orphanet:803', 'EFO:0000253'],...}
                # value: ['Orphanet:803', 'EFO:0000253']
                value = meddra_mappings[object_node["meddra"]]
                object_node["id"] = value[0]
                object_node[value[0].split(":")[0].lower()] = value[0].split(":")[1]
                if 1 < len(meddra_mappings[object_node["meddra"]]) < 3:
                    object_node[value[1].split(":")[0].lower()] = value[1].split(":")[1]
            else:
                object_node["id"] = f"MedDRA:{object_node['meddra']}"

        js_keys = [
            "sample_name",
            "method_name",
            "host_type",
            "control_name",
        ]
        association = get_association(js, js_keys)

        # n1 = subject_node["name"].split(" ")[0]
        # n2 = object_node["name"].replace(" ", "_")
        for pub in publications:
            output_dict = {
                "_id": str(uuid.uuid4().hex),  # generate random uuid
                # "edge": f"{n1}_associated_with_{n2}",
                "association": association,
                "object": object_node,
                "subject": subject_node,
                "publications": pub[js["publication_id"]],
            }

            # change the key sample_name to biospecimen_samples
            if "sample_name" in output_dict["association"]:
                output_dict["association"]["sources"] = output_dict["association"]["sample_name"]
                del output_dict["association"]["sample_name"]

            # biospecimen_samples: replace faeces to feces
            if "faeces" in association.get("sources", ""):
                association["sources"] = association["sources"].replace("faeces", "feces")

            # convert meddra, mondo and efo values to integer
            if "meddra" in output_dict["object"]:
                output_dict["object"]["meddra"] = int(output_dict["object"]["meddra"])
            if "mondo" in output_dict["object"]:
                output_dict["object"]["mondo"] = int(output_dict["object"]["mondo"])
            if "efo" in output_dict["object"]:
                output_dict["object"]["efo"] = int(output_dict["object"]["efo"])

            yield output_dict


# if __name__ == "__main__":
#     data = load_disbiome_data()
#     for obj in data:
#         print(obj)
