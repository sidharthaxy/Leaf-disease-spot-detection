"""
Stage C: combine the classifier output (which disease) with Stage B (how much area) into a
severity label.

  1. classifier confidence < min_confidence  -> "uncertain disease", use the GENERIC profile
  2. otherwise map the disease to a symptom group (configs/disease_profiles.yaml)
  3. look up that group's severity scale (configs/severity.yaml, fallback: default scale)
  4. infected % -> label
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def disease_name(label: str) -> str:
    """'Tomato___Early_blight' -> 'early blight'. Plain disease names pass through."""
    d = label.split("___", 1)[-1]
    return d.replace("_", " ").strip().lower()


def symptom_group(label: str | None, profiles: dict | None = None) -> str:
    profiles = profiles or load_yaml("disease_profiles.yaml")
    if not label:
        return "generic"
    if label in (profiles.get("overrides") or {}):
        return profiles["overrides"][label]
    name = disease_name(label)
    for rule in profiles.get("rules", []):
        if any(k in name for k in rule["keywords"]):
            return rule["group"]
    return "generic"


def get_profile(group: str, profiles: dict | None = None) -> dict:
    profiles = profiles or load_yaml("disease_profiles.yaml")
    return {"group": group, **profiles["groups"].get(group, profiles["groups"]["generic"])}


def severity_label(infected_pct: float, group: str = "generic", sev: dict | None = None) -> str:
    sev = sev or load_yaml("severity.yaml")
    scale = (sev.get("per_group") or {}).get(group, sev["default"])
    for upper, label in scale:
        if infected_pct < upper:
            return label
    return scale[-1][1]


def group_confidence(topk: list[tuple[str, float]]) -> tuple[str, float]:
    """Sum class probabilities per symptom group; return the most likely group and its probability.
    The pipeline only needs the GROUP, so 'tomato early blight 0.40 + potato early blight 0.35'
    is a confident 'necrotic' even though the exact disease is uncertain."""
    acc: dict[str, float] = {}
    for label, p in topk:
        g = symptom_group(label)
        acc[g] = acc.get(g, 0.0) + p
    g = max(acc, key=lambda k: acc[k])
    return g, acc[g]


def resolve_from_prediction(topk: list[tuple[str, float]] | None, sev: dict | None = None) -> dict:
    """topk = [(label, prob), ...] sorted by prob (pass ALL classes for the group rule).
    Returns the profile to use + uncertainty info. Rule set in severity.yaml confidence.rule:
      top1  : uncertain if p(top-1 disease) < min_confidence
      group : profile falls back to generic if p(most likely symptom group) < min_confidence
      none  : profile always follows the prediction; low confidence only flags "uncertain disease"
              (chosen on validation: MAE 17.2 vs 17.3 group rule vs 19.8 top1 rule)"""
    sev = sev or load_yaml("severity.yaml")
    min_conf = sev["confidence"]["min_confidence"]
    if not topk:
        return {"label": None, "confidence": None, "uncertain": True, "group": "generic", "topk": []}
    label, conf = topk[0]
    rule = sev["confidence"].get("rule", "top1")
    if rule == "none":
        # chosen on validation: even an uncertain group guess beats the generic profile, so the
        # profile always follows the top-1 prediction; low confidence only flags the disease name
        uncertain = conf < min_conf
        group = symptom_group(label)
    elif rule == "group":
        g, gconf = group_confidence(topk)
        uncertain = conf < min_conf          # disease name shown as uncertain...
        group = g if gconf >= min_conf else "generic"   # ...but the profile follows the confident group
    else:
        uncertain = conf < min_conf
        group = "generic" if uncertain else symptom_group(label)
    return {"label": label, "confidence": conf, "uncertain": uncertain, "group": group,
            "topk": topk[: sev["confidence"]["top_k"]]}
