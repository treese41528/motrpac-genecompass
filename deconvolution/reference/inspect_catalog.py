"""Inspect catalog — round 3. LLM analyses + unified studies entries."""
import json
import os
from pathlib import Path

# repo-relative (data/catalog is a symlink into the cluster data store)
catalog_dir = str(Path(__file__).resolve().parents[2] / "data" / "catalog")

# 1. llm_study_analysis.json -> analyses (one entry)
print("=== llm_study_analysis -> analyses (1 entry) ===")
d = json.load(open(os.path.join(catalog_dir, "llm_study_analysis.json")))
analyses = d.get("analyses", {})
print(f"type={type(analyses).__name__} len={len(analyses)}")
if isinstance(analyses, dict):
    fk = list(analyses.keys())[0]
    print(f"first key: {fk}")
    print(json.dumps(analyses[fk], indent=2, default=str)[:2500])
elif isinstance(analyses, list):
    print(json.dumps(analyses[0], indent=2, default=str)[:2500])

# 2. unified_studies.json -> studies (one entry)
print("\n=== unified_studies -> studies (1 entry) ===")
d2 = json.load(open(os.path.join(catalog_dir, "unified_studies.json")))
studies = d2.get("studies", {})
print(f"type={type(studies).__name__} len={len(studies)}")
if isinstance(studies, dict):
    fk = list(studies.keys())[0]
    print(f"first key: {fk}")
    print(json.dumps(studies[fk], indent=2, default=str)[:2500])
elif isinstance(studies, list):
    print(json.dumps(studies[0], indent=2, default=str)[:2500])

# 3. master_catalog tissues field — sample of distinct tissue values
print("\n=== master_catalog -> distinct tissue values (sample) ===")
d3 = json.load(open(os.path.join(catalog_dir, "master_catalog.json")))
mc_studies = d3.get("studies", [])
all_tissues = set()
for s in mc_studies:
    for t in s.get("tissues", []):
        all_tissues.add(t)
print(f"n_unique_tissues={len(all_tissues)}")
for t in sorted(all_tissues)[:40]:
    print(f"  {t}")
