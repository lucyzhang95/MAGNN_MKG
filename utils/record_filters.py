#


def is_organism_and_disease(rec):
    return (
        rec.get("association", {}).get("predicate")
        == "OrganismalEntityAsAModelOfDiseaseAssociation"
    )


def is_small_molecule_and_taxid(rec):
    return rec.get("object", {}).get(
        "type"
    ) == "biolink:SmallMolecule" and "taxid" in rec.get("subject", {})


def is_small_molecule_and_gene(rec):
    return (
        rec.get("subject", {}).get("type") == "biolink:SmallMolecule"
        and rec.get("object", {}).get("type") == "biolink:Gene"
    )


def is_not_pubchem_cid(rec, node):
    return (
        ":" in rec[node]["id"]
        and "PUBCHEM.COMPOUND" not in rec[node]["id"]
    )


def is_not_id(rec, node):
    return ":" in rec[node]["id"]


def is_uniprotkb(rec):
    return rec["object"]["id"] and "UniProtKG" in rec["object"]["id"]
