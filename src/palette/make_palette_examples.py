"""
Add ornamental (palette-branch) examples to the demo gallery: held-out TEST photos only.

    python -m src.palette.make_palette_examples --data data/router
    python -m src.palette.make_palette_examples --species bloodleaf,red_cordyline --fetch   # add species, keep the rest

--fetch re-downloads only the candidate test photos it needs (when data/router was deleted), using the
observation id from the router manifest, and accepts a photo only if its dHash matches the manifest (same photo).

Selection (no cherry-picking on the result): for each listed species, a random test photo (fixed seed) with
a licence that allows non-commercial redistribution of resized copies (CC0 / BY / BY-SA / BY-NC / BY-NC-SA) that the router actually sends to the palette branch.
Plus synthetic-damage copies of two of them (real PlantSeg VAL lesions pasted on, clearly labelled).
Appends rows (kind = "palette") to demo/examples/manifest.csv and the attribution to ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.palette.species import display
from src.palette.synthetic import build_lesion_bank, paste_lesions, rough_leaf_prior

ROOT = Path(__file__).resolve().parents[2]
DST = ROOT / "demo" / "examples"
SPECIES = ["croton", "coleus", "caladium", "purple_heart", "copperleaf", "polka_dot", "golden_pothos", "dieffenbachia",
           "bloodleaf", "red_cordyline"]
DAMAGED = ["croton", "golden_pothos"]
OK_LICENCES = {"cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa"}   # no -nd: we resize the photos
LIC_NAME = {"cc0": "CC0 1.0", "cc-by": "CC BY 4.0", "cc-by-sa": "CC BY-SA 4.0", "cc-by-nc": "CC BY-NC 4.0",
            "cc-by-nc-sa": "CC BY-NC-SA 4.0"}


def fetch_photo(r, dst: Path) -> bool:
    """Re-download one router-manifest photo from iNaturalist; keep it only if its dHash matches the manifest."""
    import json
    import time
    import urllib.request

    from src.router.collect_router_data import UA, save_resized

    obs_id, photo_id = Path(r.image_path).stem.split("_")[1:3]
    with urllib.request.urlopen(urllib.request.Request(f"https://api.inaturalist.org/v1/observations/{obs_id}", headers=UA), timeout=60) as f:
        obs = json.load(f)["results"]
    time.sleep(1.1)                                            # iNaturalist asks for <= 1 request / second
    ph = next((p for o in obs for p in o.get("photos", []) if str(p["id"]) == photo_id), None)
    if ph is None:
        return False
    url = ph["url"].replace("/square.", "/medium.")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as f:
        h = save_resized(f.read(), dst)
    time.sleep(1.1)
    if h is None or bin(h ^ int(r.dhash)).count("1") > 4:     # not the same photo any more
        dst.unlink(missing_ok=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "router"))
    ap.add_argument("--plantseg", default=str(ROOT / "data" / "dev_val"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--species", help="comma-separated: add only these species and keep the other palette examples")
    ap.add_argument("--fetch", action="store_true", help="re-download missing candidate photos from iNaturalist")
    a = ap.parse_args()
    data = Path(a.data)
    man = pd.read_csv(data / "manifest.csv" if (data / "manifest.csv").exists() else ROOT / "manifests" / "router_manifest.csv")
    species = a.species.split(",") if a.species else SPECIES
    ex = pd.read_csv(DST / "manifest.csv", dtype=str, keep_default_na=False)   # keep existing rows as text
    if "kind" in ex:
        drop = ex.kind == "palette"
        if a.species:                                          # only replace the species being (re)added
            drop &= ex.label.isin([f"{display(sp)['common']}___{t}" for sp in species for t in ("healthy", "synthetic_damage")])
        ex = ex[~drop]
    from src.imageproc.plant_check import check_plant
    from src.imageproc.preprocess import preprocess
    from src.router.router import Router

    router = Router()
    rng = np.random.default_rng(a.seed)
    bank = build_lesion_bank(a.plantseg, seed=a.seed)
    rows, credits = [], []
    next_id = int(ex.example_id.astype(int).max()) + 1
    (DST / "palette").mkdir(parents=True, exist_ok=True)
    for sp in species:
        cand = man[(man.label == sp) & (man.split == "test") & man.licence.isin(OK_LICENCES)]
        cand = cand.sample(frac=1, random_state=a.seed)
        for r in cand.itertuples():
            src = data / r.image_path
            if not src.exists() and a.fetch and not fetch_photo(r, src):
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue
            gate = check_plant(preprocess(img)[0])
            route = router.route(img, gate["green_pct"])
            if route["route"] != "palette" or route["species"] != sp:
                continue
            name = display(sp)["common"]
            rel = f"palette/{Path(r.image_path).name}"
            cv2.imwrite(str(DST / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            base = {"image_path": rel, "mask_path": "", "plant": name, "split": "test", "source": "inaturalist",
                    "kind": "palette", "licence": LIC_NAME[r.licence], "attribution": r.attribution, "url": r.url}
            rows.append({**base, "example_id": next_id, "disease": "healthy (as photographed)", "label": f"{name}___healthy", "note": ""})
            next_id += 1
            credits.append(f"- `{rel}`: {r.attribution}, {LIC_NAME[r.licence]}, {r.url}")
            if sp in DAMAGED and r.licence in ("cc0", "cc-by", "cc-by-nc"):  # SA + NC lesions would conflict
                pre = preprocess(img, {"clahe": False})[0]
                dmg, _ = paste_lesions(pre, rough_leaf_prior(pre), bank, rng, n_range=(3, 5))
                rel2 = f"palette/synthetic_{Path(r.image_path).name}"
                cv2.imwrite(str(DST / rel2), dmg, [cv2.IMWRITE_JPEG_QUALITY, 90])
                rows.append({**base, "image_path": rel2, "example_id": next_id, "disease": "synthetic damage",
                             "label": f"{name}___synthetic_damage",
                             "note": "SYNTHETIC: real PlantSeg val lesions (CC BY-NC 4.0) pasted onto this photo"})
                next_id += 1
                credits.append(f"- `{rel2}`: modified from the photo above (PlantSeg lesions pasted on, CC BY-NC 4.0 parts)")
            break
        else:
            print(f"[skip] {sp}: no permissive-licence test photo routed to the palette branch", flush=True)
    out = pd.concat([ex, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(DST / "manifest.csv", index=False)
    att = DST / "ATTRIBUTION.md"
    head, _, old = att.read_text().partition("\n## Ornamental examples")
    kept = set(ex.image_path) if a.species else set()          # credits of the examples we keep
    credits = [ln for ln in old.splitlines() if ln.startswith("- `") and ln[3:].split("`")[0] in kept] + credits
    text = head.rstrip() + "\n"
    text += ("\n## Ornamental examples (palette branch)\n\n"
             "Held-out TEST photos from iNaturalist observations (never used to fit palettes or train the router), "
             "resized to a 384 px long side. Licence and author per file (as given by iNaturalist):\n\n" + "\n".join(credits) + "\n")
    att.write_text(text)
    print(f"added {len(rows)} palette examples")


if __name__ == "__main__":
    main()
