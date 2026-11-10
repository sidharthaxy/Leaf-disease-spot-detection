"""Display names of the palette species (router class -> common name, scientific name, kind)."""

SPECIES = {
    "croton": ("Croton", "Codiaeum variegatum", "multicoloured"),
    "coleus": ("Coleus", "Coleus scutellarioides (syn. Plectranthus scutellarioides)", "multicoloured"),
    "caladium": ("Caladium", "Caladium bicolor", "multicoloured"),
    "purple_heart": ("Purple heart", "Tradescantia pallida", "purple"),
    "copperleaf": ("Copperleaf", "Acalypha wilkesiana", "red / copper"),
    "polka_dot": ("Polka dot plant", "Hypoestes phyllostachya", "spotted pink / red"),
    "bloodleaf": ("Bloodleaf", "Iresine diffusa f. herbstii", "red / purple"),
    "red_cordyline": ("Red cordyline", "Cordyline fruticosa", "red / purple"),
    "golden_pothos": ("Golden pothos", "Epipremnum aureum", "variegated green"),
    "dieffenbachia": ("Dieffenbachia", "Dieffenbachia seguine", "variegated green"),
}


def display(cls: str) -> dict:
    common, latin, kind = SPECIES.get(cls, (cls.replace("_", " ").capitalize(), "", ""))
    return {"id": cls, "common": common, "latin": latin, "kind": kind}
