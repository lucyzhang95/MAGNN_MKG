import os
import pathlib
import pickle

# import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import biothings_client


def strip_tag_namespace(tag: str) -> str:
    """Strip namespace from main element tag
    Each element tag from HMDB contains "http://www.hmdb.ca" as namespace
    e.g. {http://www.hmdb.ca}accession
    Remove the namespace for easier data processing

    :param tag: element tags (e.g. <accession>HMDB0000001</accession>)
    :return: original tag without namespace
    """
    idx = tag.rfind("}")
    # rfind() method "not found" == -1
    if idx != -1:  # if idx is not "not found"
        tag = tag[idx + 1 :]
    return tag


def get_all_microbe_names(input_xml: str | pathlib.Path) -> Iterator[str]:
    """Obtain the names of microbial species associated with each metabolite
    Microbe names are under the term//////root/ontology tag
    Brief structure of the input_xml file
    <?xml version="1.0" encoding="UTF-8"?>
    <hmdb xmlns="http://www.hmdb.ca">
    <metabolite>
        <ontology>
            <root>...</root>
            <root>
                <term>Disposition</term>
                    <descendants>
                        <descendant>...</descendant>
                        <descendant>
                            <term>Microbe</term>
                            <descendants>
                                <descendant>
                                    <term>Escherichia</term>
                                    <descendants>
                                        <descendant>
                                            <term>Escherichia coli</term>
                                        </descendant>
                                    </descendants>
                                </descendant>
                            </descendants>
                        </descendant>
                    </descendants>
            </root>
            <root>...</root>
            <root>...</root>
        </ontology>
    </metabolite>
    </hmdb>

    :param input_xml: hmdb_metabolites.xml file
    :return: an iterator of strings of microbial names
    """
    if not os.path.exists("data/hmdb_mapped_taxon.pkl"):
        for event, elem in ET.iterparse(input_xml, events=("start", "end")):
            if event == "end" and elem.tag.endswith("metabolite"):
                for metabolite in elem:
                    tagname = strip_tag_namespace(metabolite.tag)
                    if tagname == "ontology":
                        for descendant in metabolite.iter("{http://www.hmdb.ca}descendant"):
                            term = descendant.findall("{http://www.hmdb.ca}term")
                            if term and term[0].text == "Microbe":
                                # findall returns a list of items
                                microbe_descendants = descendant.findall(
                                    ".//{http://www.hmdb.ca}term"
                                )
                                # The first item is always "Microbe", so skip it
                                for microbe_name in microbe_descendants[1:]:
                                    yield microbe_name.text.strip()


def get_taxon_info(microbial_names: set) -> dict:
    """Use biothings_client to query the microbe from hmdb
    get ncbi taxon_id and associated info per microbe
    Query result can be more than 1, filtration is performed
    Filtered by:
    1. bacteria kingdom (taxid:2),
    2. must be either genus or species
    3. highest score associated with the query
    4. same score, get the first record

    :param microbial_names: a set of microbial scientific names
    :return: a dictionary contains all taxid and associated info
    """
    t = biothings_client.get_client("taxon")
    if not os.path.exists("data/hmdb_mapped_taxon.pkl"):
        taxon_info = t.querymany(
            microbial_names,
            scopes="scientific_name",
            fields=[
                "_id",
                "scientific_name",
                "lineage",
                "parent_taxid",
                "rank",
            ],
        )
        # print(taxon_info)

        unique_taxon_d = {}
        taxon_d = defaultdict(list)
        for d in taxon_info:
            if "notfound" not in d:
                if d["rank"] != "subgenus":
                    taxon_d[d["query"]].append((d["_score"]))

        # Take the highest score associated with the query microbial name
        max_score = dict([(name, max(score)) for name, score in taxon_d.items()])
        # print(max_score)

        for d in taxon_info:
            if d["query"] in max_score and d["_score"] == max_score[d["query"]]:
                unique_taxon_d[d["query"]] = {
                    "taxid": int(d["_id"]),
                    "scientific_name": d["scientific_name"],
                    "lineage": d["lineage"],
                    "parent_taxid": d["parent_taxid"],
                    "rank": d["rank"],
                }
        print(unique_taxon_d)
        yield unique_taxon_d


def save_mapped_taxon_to_pkl(input_xml, output_pkl: str | Path):
    """save the mapped microbial ncbi taxon info to pickle file

    :param input_xml: hmdb_metabolites.xml
    :param output_pkl: hmdb_mapped_taxon.pkl
    :return: hmdb_mapped_taxon.pkl
    """
    microbes_l = []
    microbes = get_all_microbe_names(input_xml)
    for microbe in microbes:
        microbes_l.append(microbe)

    unique_microbes = set(microbes_l)
    taxon = get_taxon_info(unique_microbes)

    check_pkl_file = os.path.exists(output_pkl)
    if not check_pkl_file:
        for taxon_d in taxon:
            with open(output_pkl, "wb") as handle:
                pickle.dump(taxon_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
                return output_pkl
    else:
        pass


def remove_empty_values(new_association_list: list) -> list:
    """remove the empty values associated with pathways, proteins and diseases dictionaries

    :param new_association_list: list of pathways, proteins and diseases
    :return: a new list of pathways, proteins and diseases with no empty values
    """
    filtered_association = []
    if new_association_list:
        for d in new_association_list:
            if any(d.values()):
                filtered_association.append({k: v for k, v in d.items() if v})
        new_association_list = filtered_association
        return new_association_list


def remove_duplicate_microbe(microbe_list: list) -> list:
    """remove the duplicated genus term for the microbes that have both genus and species
    Due to the way how xml file is processed by ET, the same metabolite will have duplications
    e.g.
    ...
    <descendants>
        <descendant>
            <term>Escherichia</term>
            <descendants>
                <descendant>
                    <term>Escherichia coli</term>
    Output will be ["Escherichia", "Escherichia coli"]
    This function only keeps the genus term "Escherichia coli"

    :param microbe_list: a list of microbes that have both genus and species
    :return: a list of microbes with genus term
    """
    unique_microbes_l = []
    for microbe in microbe_list:
        is_unique = True
        for other_microbe in microbe_list:
            if microbe != other_microbe and microbe in other_microbe:
                is_unique = False
                break
        if is_unique:
            unique_microbes_l.append(microbe)
    return unique_microbes_l


def replace_dict_keys(d: dict, new_k: str, old_k: str):
    """replace old dictionary key with new standardized dictionary key
    and delete the old dictionary

    :param d: output_dict
    :param new_k: "kegg_compound", "chebi", "bigg", etc.
    :param old_k: "kegg_id", "chebi_id", "bigg_id", etc.
    :return: updated output dictionary
    """
    if "xrefs" in d and old_k in d["xrefs"]:
        d["xrefs"][new_k] = d["xrefs"].get(old_k)
        del d["xrefs"][old_k]


def add_biolink_type(output_d: dict, output_key: str, biolink_type: str):
    """adds a Biolink model type to each dictionary in a list within a given key of an output dictionary.
    This function checks if a specified key exists in the provided dictionary and
    for each dictionary in the list, the function adds a new key-value pair provided Biolink type

    :param output_d: output dict
    :param output_key: output dictionary entity key
    :param biolink_type: biolink model type
    :return: None
    """
    if output_key in output_d and isinstance(output_d[output_key], list):
        for d in output_d[output_key]:
            d["type"] = biolink_type


def load_hmdb_data() -> Iterator[dict]:
    """load data from HMDB database

    :param data_path: path required for biothings_cli data upload (not used if run locally)
    :return: a generator consists of an iterator of dictionaries containing hmdb accession id,
    metabolite name, chemical_formula, description, cross-reference ontologies, associated_microbes,
    pathways, diseases, and proteins, state, status, anatomical locations
    """
    path = Path.cwd()
    file_path = os.path.join(path, "source", "hmdb_metabolites.zip")
    file_name = "hmdb_metabolites.xml"

    with zipfile.ZipFile(file_path, "r") as zip_f:
        with zip_f.open(file_name) as xml_f:
            save_mapped_taxon_to_pkl(xml_f, "data/hmdb_mapped_taxon.pkl")
            for event, elem in ET.iterparse(xml_f, events=("start", "end")):
                if event == "end" and elem.tag.endswith("metabolite"):
                    output = {
                        "_id": None,
                        "name": None,
                        "chemical_formula": None,
                        "description": None,
                        "xrefs": {},
                        "associated_microbes": [],
                        "associated_pathways": [],
                        "associated_diseases": [],
                        "associated_proteins": [],
                    }
                    for metabolite in elem:
                        tname = strip_tag_namespace(metabolite.tag)
                        tname_list = [
                            "chebi_id",
                            "chemspider_id",
                            "drugbank_id",
                            "foodb_id",
                            "smiles",
                            "inchikey",
                            "pdb_id",
                        ]
                        if tname == "accession":
                            if metabolite.text:
                                output["_id"] = metabolite.text
                        elif tname == "name":
                            if metabolite.text:
                                output["name"] = metabolite.text.lower()
                        elif tname in tname_list:
                            if metabolite.text:
                                if "_id" in tname:
                                    prefix = tname.replace("_id", "").upper()
                                    xref_ids = {tname: f"{prefix}:{metabolite.text}"}
                                else:
                                    xref_ids = {tname: metabolite.text}
                                output["xrefs"].update(xref_ids)
                        elif tname == "pubchem_compound_id":
                            if metabolite.text:
                                output["xrefs"].update(
                                    {"pubchem_cid": f"PUBCHEM.COMPOUND:{int(metabolite.text)}"}
                                )
                        elif tname == "kegg_id":
                            if metabolite.text:
                                output["xrefs"].update(
                                    {"kegg_compound": f"KEGG.COMPOUND:{metabolite.text}"}
                                )
                        elif tname == "status":
                            if metabolite.text:
                                output["status"] = metabolite.text
                        elif tname == "description":
                            if metabolite.text:
                                output["description"] = metabolite.text
                        elif tname == "state":
                            if metabolite.text:
                                output["state"] = metabolite.text.lower()
                        elif tname == "chemical_formula":
                            if metabolite.text:
                                output["chemical_formula"] = metabolite.text
                        elif tname == "average_molecular_weight":
                            if metabolite.text:
                                output["average_mw"] = float(metabolite.text)
                        elif tname == "monisotopic_molecular_weight":
                            if metabolite.text:
                                output["monoisotopic_mw"] = float(metabolite.text)
                        elif tname == "ontology":
                            for descendant in metabolite.iter("{http://www.hmdb.ca}descendant"):
                                term = descendant.findall("{http://www.hmdb.ca}term")
                                if term and term[0].text == "Microbe":
                                    microbe_descendants = descendant.findall(
                                        ".//{http://www.hmdb.ca}term"
                                    )
                                    # The output of microbe_descendants = ['Microbe', 'Alcaligenes', 'Alcaligenes eutrophus']
                                    microbe_names = [
                                        microbe.text for microbe in microbe_descendants[1:]
                                    ]
                                    unique_microbes = remove_duplicate_microbe(microbe_names)
                                    with open("data/hmdb_mapped_taxon.pkl", "rb") as handle:
                                        taxon = pickle.load(handle)
                                        for microbe in unique_microbes:
                                            if microbe in taxon:
                                                output[microbe] = output[
                                                    "associated_microbes"
                                                ].append(taxon[microbe])
                                            else:
                                                output[microbe] = output[
                                                    "associated_microbes"
                                                ].append({"scientific_name": microbe.lower()})
                        elif tname == "biological_properties":
                            for child in metabolite.iter(
                                "{http://www.hmdb.ca}biological_properties"
                            ):
                                biospec_loc = child.find(
                                    "{http://www.hmdb.ca}biospecimen_locations"
                                )
                                if biospec_loc:
                                    specimens = [specimen.text.lower() for specimen in biospec_loc]
                                    output["sources"] = specimens

                                pathways = child.find("{http://www.hmdb.ca}pathways")
                                if pathways:
                                    for pathway in pathways.iter("{http://www.hmdb.ca}pathway"):
                                        pathway_dict = {
                                            "name": pathway.findtext(
                                                "{http://www.hmdb.ca}name"
                                            ).lower(),
                                            "kegg_map_id": pathway.findtext(
                                                "{http://www.hmdb.ca}kegg_map_id"
                                            ),
                                            "smpdb_id": pathway.findtext(
                                                "{http://www.hmdb.ca}smpdb_id"
                                            ),
                                        }
                                        output["associated_pathways"].append(pathway_dict)
                        elif tname == "diseases":
                            for diseases in metabolite.iter("{http://www.hmdb.ca}disease"):
                                if diseases:
                                    disease_dict = {
                                        "name": diseases.findtext(
                                            "{http://www.hmdb.ca}name"
                                        ).lower()
                                    }
                                    if diseases.findtext("{http://www.hmdb.ca}omim_id"):
                                        disease_dict[
                                            "omim"
                                        ] = f"OMIM:{diseases.findtext('{http://www.hmdb.ca}omim_id')}"
                                    disease_dict["pmid"] = []
                                    for ref in diseases.findall(".//{http://www.hmdb.ca}pubmed_id"):
                                        if ref.text:
                                            disease_dict["pmid"].append(int(ref.text))
                                    output["associated_diseases"].append(disease_dict)
                        elif tname == "protein_associations":
                            for proteins in metabolite.iter("{http://www.hmdb.ca}protein"):
                                if proteins:
                                    protein_dict = {
                                        "name": proteins.findtext(
                                            "{http://www.hmdb.ca}name"
                                        ).lower(),
                                        "uniprotkb": proteins.findtext(
                                            "{http://www.hmdb.ca}uniprot_id"
                                        ),
                                    }
                                    output["associated_proteins"].append(protein_dict)

                    output["associated_pathways"] = remove_empty_values(
                        output["associated_pathways"]
                    )
                    output["associated_diseases"] = remove_empty_values(
                        output["associated_diseases"]
                    )
                    output["associated_proteins"] = remove_empty_values(
                        output["associated_proteins"]
                    )
                    output = {k: v for k, v in output.items() if v}
                    replace_dict_keys(output, "chebi", "chebi_id")
                    replace_dict_keys(output, "chemspider", "chemspider_id")
                    replace_dict_keys(output, "drugbank", "drugbank_id")
                    replace_dict_keys(output, "foodb", "foodb_id")
                    replace_dict_keys(output, "pdb", "pdb_id")

                    # assign microbial type following biolink schema
                    if "associated_microbes" in output and isinstance(
                        output["associated_microbes"], list
                    ):
                        for taxon_dict in output["associated_microbes"]:
                            if "lineage" in taxon_dict:
                                if 2 in taxon_dict["lineage"]:
                                    taxon_dict["type"] = "biolink:Bacterium"
                                elif 10239 in taxon_dict["lineage"]:
                                    taxon_dict["type"] = "biolink:Virus"
                                elif 4751 in taxon_dict["lineage"]:
                                    taxon_dict["type"] = "biolink:Fungus"
                                elif 2157 in taxon_dict["lineage"]:
                                    taxon_dict["type"] = "biolink:Archaea"
                                else:
                                    taxon_dict["type"] = "biolink:OrganismalEntity"

                    # add biolink type to associated_pathways
                    add_biolink_type(output, "associated_pathways", "biolink:Pathway")
                    # add disease type to associated_diseases
                    add_biolink_type(output, "associated_diseases", "biolink:Disease")
                    # add biolink type to associated_proteins
                    add_biolink_type(output, "associated_proteins", "biolink:Protein")

                    yield output


# if __name__ == "__main__":
#     hmdb_data = load_hmdb_data()
    # examp = [print(obj) for obj in hmdb_data if obj["_id"] == "HMDB0000020"]
    # print(examp)
    # parser_o = [obj for obj in hmdb_data if "associated_microbes" in obj]
    # print(parser_o)
    # print(len(set(parser_o)))
    # with open("data/hmdb_microbe_metabolites.pkl", "wb") as handle:
    #     pickle.dump(parser_o, handle, protocol=pickle.HIGHEST_PROTOCOL)
