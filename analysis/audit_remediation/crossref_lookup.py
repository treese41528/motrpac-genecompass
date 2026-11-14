#!/usr/bin/env python
"""crossref_lookup.py -- print the Crossref record of each DOI, so every reference cited in the remediation
documents is checked against the publisher's metadata before it is written down.

Usage: crossref_lookup.py <out_tsv> key=doi [key=doi ...]     (appends; a generic user agent, no contact details)
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

out = Path(sys.argv[1])
cols = ["key", "doi", "first_authors", "n_authors", "title", "journal", "year_issued", "volume", "issue", "pages", "type"]
new = not out.exists()
with open(out, "a") as fh:
    if new:
        fh.write("\t".join(cols) + "\n")
    for spec in sys.argv[2:]:
        key, doi = spec.split("=", 1)
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "reference-check/1.0"})
            m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
            au = m.get("author", [])
            names = [f"{a.get('family', a.get('name', '?'))} {''.join(w[0] for w in a.get('given', '').split() if w)}".strip()
                     for a in au[:4]]
            row = [key, doi, ", ".join(names) + (", et al." if len(au) > 4 else ""), str(len(au)),
                   " ".join((m.get("title") or [""])[0].split()), (m.get("container-title") or [""])[0],
                   str((m.get("issued", {}).get("date-parts") or [[""]])[0][0]), m.get("volume", ""),
                   m.get("issue", ""), m.get("page", ""), m.get("type", "")]
        except Exception as e:                       # noqa: BLE001
            row = [key, doi, f"LOOKUP FAILED: {e}"] + [""] * 8
        fh.write("\t".join(row) + "\n")
        print(" | ".join(row))
        time.sleep(0.4)
