#
def is_small_molecule_and_taxid(rec):
    return rec.get("object", {}).get(
        "type"
    ) == "biolink:SmallMolecule" and "taxid" in rec.get("subject", {})


def is_small_molecule_and_gene(rec):
    return (
        rec.get("subject", {}).get("type") == "biolink:SmallMolecule"
        and rec.get("object", {}).get("type") == "biolink:Gene"
    )
