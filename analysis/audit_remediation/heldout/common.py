"""common.py -- paths and small helpers shared by the held-out test scripts (STEP5_PREREG.md M2, STEP4_PREREG.md V3).

Nothing here writes into the training tree. Every output of the held-out test goes under
  data/audit_remediation/heldout/                      (QC h5ad files, tokenized cells, evaluator output)
  docs/audit/remediation/record/step4_heldout/         (small tables)

The steps, in order (each script's docstring says what it implements; run from the repository root with
PIPELINE_PROFILE unset and TMPDIR under the project's tmp/):
  01_series_list.py       the series the catalog merge dropped; series-level exclusions        (login node, seconds)
  02_cell_qc.py           Stage 2 cell QC of the remaining series -> heldout/qc_h5ad/           (batch job)
  03_file_plan.py         file-level plan (non-rat, not raw counts, copies); QC count tables    (batch job)
  04_tokenize.py          Stage 5 tokenization as v2 -> heldout/tokenized/; the 2,000-cell draw (batch job)
  05_check_unseen.py      supplementary: no evaluation cell's token sequence equals a training cell's (batch job)
  05b_check_barcodes.py   supplementary: no kept file shares its cells with a training QC file  (batch job)
  eval_mlm.py eval --by-tier --kind rat --model <v2|April> --probe data/training/eval_probe_v2
                          the unchanged C1 evaluator on the in-sample probe (reproduces the record first)
  06_eval.py              the same evaluator per cell, on the probe and on the evaluation set   (GPU job)
  07_tables.py            loss tables and the registered ratio and reading                      (login node, seconds)
"""
import csv
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.setdefault("PIPELINE_ROOT", str(Path(__file__).resolve().parents[3])))
HELD = ROOT / "data/audit_remediation/heldout"
REC = ROOT / "docs/audit/remediation/record/step4_heldout"
GEO = ROOT / "data/raw/geo/single_cell/geo_datasets"
CATALOG = ROOT / "data/catalog"
PLAN = ROOT / "config/corpus_v2_plan.tsv"
INVENTORY = ROOT / "reports/annotations/annotation_inventory.tsv"
COVERAGE = ROOT / "data/training/gene_universe/study_coverage.tsv"

QC_DIR = HELD / "qc_h5ad"
QC_REPORTS = HELD / "qc_reports"
TOK_DIR = HELD / "tokenized"
EVAL_DIR = HELD / "eval"

# Forbidden output roots (the task's hard rule); out_path() refuses a path under any of them.
PROTECTED = [ROOT / p for p in ("data/training", "data/catalog", "data/models", "data/deconvolution",
                                "data/results", "reports")]

GSM_RE = re.compile(r"(GSM\d{6,9})")
REL_RE = re.compile(r"(SuperSeries of|SubSeries of):\s*(GSE\d+)")
SPATIAL_RE = re.compile(r"visium|spatial transcriptom|slide-?seq|stereo-?seq", re.I)
RAT = "Rattus norvegicus"
DRAW_SEED = 20261114          # evaluation-set draw (2,000 cells per study); the registration date
CAP_PER_STUDY = 2000          # STEP4_PREREG.md V3


def out_path(p):
    """Return p as a Path after checking it is not inside a protected (training / results) folder."""
    p = Path(p).resolve()
    for q in PROTECTED:
        q = q.resolve()
        if p == q or q in p.parents:
            raise RuntimeError(f"refusing to write under {q}: {p}")
    return p


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))


def write_tsv(path, rows, fields=None):
    path = out_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_soft(accession, geo_dir=GEO):
    """Read <GSE>_family.soft.gz as harvested. None when the file is absent.

    Returns {title, gsm: [..], relations: [(kind, GSE)], organisms: {gsm: [organism, ..]}, sample_title: {gsm: ..}}.
    """
    p = Path(geo_dir) / accession / f"{accession}_family.soft.gz"
    if not p.is_file():
        return None
    out = {"title": "", "gsm": [], "relations": [], "organisms": {}, "sample_title": {}, "source_name": {},
           "characteristics": {}, "library_strategy": {}, "spatial": {}}
    cur = None
    with gzip.open(p, "rt", errors="replace") as f:
        for line in f:
            if line[0] not in "!^":
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key == "^SAMPLE":
                cur = val
            elif key == "!Series_title":
                out["title"] = val
            elif key == "!Series_sample_id":
                out["gsm"].append(val)
            elif key == "!Series_relation":
                m = REL_RE.search(val)
                if m:
                    out["relations"].append((m.group(1), m.group(2)))
            elif cur and key.startswith("!Sample_organism_ch"):
                out["organisms"].setdefault(cur, []).append(val)
            elif cur and key == "!Sample_title":
                out["sample_title"][cur] = val
            elif cur and key == "!Sample_source_name_ch1":
                out["source_name"][cur] = val
            elif cur and key == "!Sample_characteristics_ch1":
                out["characteristics"].setdefault(cur, []).append(val)
            elif cur and key == "!Sample_library_strategy":
                out["library_strategy"][cur] = val
            if cur and (key in ("!Sample_title", "!Sample_source_name_ch1", "!Sample_characteristics_ch1",
                                "!Sample_library_strategy")
                        or (key == "!Sample_data_processing" and val.lower().startswith("library strategy"))):
                if SPATIAL_RE.search(val):
                    out["spatial"][cur] = True       # a spatial assay (spots, not cells), by the sample's own record
    return out


def sample_is_rat(organisms):
    """The corpus rule (config/corpus_v2_plan.tsv, action drop_nonrat): a file is kept only when the GEO
    organism of its sample is Rattus norvegicus. Every kept corpus file with a GSM has exactly that one
    organism, so a sample that lists a second organism (a chimera or a mixed-species sample) is not rat."""
    return list(organisms) == [RAT]


def training_gsms(plan_rows=None, geo_dir=GEO):
    """Every GEO sample accession that can be tied to a series of the corpus plan, with where it was found.

    Sources, united (the widest reading of 'a training file's GSM'): the plan's gsm column, the corpus
    inventory's gsm column, and for each of the plan's series the sample list of its SOFT file, its
    <GSE>_samples.tsv and the GSM prefixes of the file names in its folder. Returns {gsm: {(series, source)}}.
    """
    plan_rows = plan_rows if plan_rows is not None else read_tsv(PLAN)
    found = {}

    def add(gsm, series, src):
        if gsm:
            found.setdefault(gsm, set()).add((series, src))

    for r in plan_rows:
        add(r.get("gsm", ""), r["study"], "plan")
    if INVENTORY.is_file():
        for r in read_tsv(INVENTORY):
            add(r.get("gsm", ""), r.get("accession", ""), "inventory")
    for series in sorted({r["study"] for r in plan_rows}):
        soft = parse_soft(series, geo_dir)
        for g in (soft or {}).get("gsm", []):
            add(g, series, "soft")
        d = Path(geo_dir) / series
        st = d / f"{series}_samples.tsv"
        if st.is_file():
            with open(st, errors="replace") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    g = (row.get("geo_accession") or "").strip()
                    if g.startswith("GSM"):
                        add(g, series, "samples_tsv")
        if d.is_dir():
            for _, _, files in os.walk(d):
                for fn in files:
                    m = GSM_RE.search(fn)
                    if m:
                        add(m.group(1), series, "filename")
    return found


def series_exclusions(candidates, soft, train_series, train_gsm):
    """The registered series-level exclusions (STEP5_PREREG.md M2 'Check before use'; STEP4_PREREG.md V3).

    candidates: accessions; soft: {accession: parse_soft() result or None} for candidates and training series;
    train_series: set of the corpus plan's series; train_gsm: {gsm: {(series, source)}}.
    Returns (excluded {accession: [reason rows]}, notes {accession: [note rows]}). A note is not an exclusion.
    """
    excluded, notes = {}, {}
    # relations declared by training series, turned round: (candidate, kind seen from the candidate, training series)
    declared_by_training = []
    for t in sorted(train_series):
        for kind, other in (soft.get(t) or {}).get("relations", []):
            declared_by_training.append((other, "SubSeries of" if kind == "SuperSeries of" else "SuperSeries of", t))
    # super-series of each training series (for the sibling note)
    train_supers = {}
    for t in sorted(train_series):
        for kind, other in (soft.get(t) or {}).get("relations", []):
            if kind == "SubSeries of":
                train_supers.setdefault(other, set()).add(t)

    for a in candidates:
        s = soft.get(a)
        rows = []
        if s is None:
            notes.setdefault(a, []).append({"rule": "no_soft_file",
                                            "detail": "no family.soft.gz on disk: samples and relations unknown"})
            continue
        shared = sorted(g for g in s["gsm"] if g in train_gsm)
        if shared:
            ser = sorted({x[0] for g in shared for x in train_gsm[g]})
            rows.append({"rule": "shares_gsm_with_training",
                         "detail": f"{len(shared)} of {len(s['gsm'])} samples are samples of training series "
                                   f"{','.join(ser)} (e.g. {','.join(shared[:3])})"})
        rel = {(k, o) for k, o in s["relations"] if o in train_series}
        rel |= {(k, t) for c, k, t in declared_by_training if c == a}
        for kind, other in sorted(rel):
            rows.append({"rule": "superseries_of_training" if kind == "SuperSeries of" else "subseries_of_training",
                         "detail": f"{kind} {other}"})
        if rows:
            excluded[a] = rows
        sib = sorted({t for k, o in s["relations"] if k == "SubSeries of" for t in train_supers.get(o, ())})
        if sib:
            sup = sorted({o for k, o in s["relations"] if k == "SubSeries of" and o in train_supers})
            notes.setdefault(a, []).append({"rule": "sibling_of_training_under_superseries",
                                            "detail": f"sub-series of {','.join(sup)}, which is also the "
                                                      f"super-series of training series {','.join(sib)}"})
    return excluded, notes


def draw_cells(n_cells, cap=CAP_PER_STUDY, seed=DRAW_SEED, study=""):
    """Sorted row indices of the cells drawn for one study: all of them when n_cells <= cap, otherwise `cap`
    without replacement from numpy.random.default_rng([seed, crc32(study)])."""
    import zlib
    import numpy as np
    if n_cells <= cap:
        return np.arange(n_cells)
    rng = np.random.default_rng([seed, zlib.crc32(study.encode())])
    return np.sort(rng.choice(n_cells, size=cap, replace=False))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def counter_str(values):
    c = Counter(values)
    return "; ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))))


# ---------------------------------------------------------------------------------------------------------
# What genome a source matrix is annotated with, read from its own feature names (no cell is read).
# ---------------------------------------------------------------------------------------------------------
_MOUSE_SYM = re.compile(r"^(Gm\d+|[0-9A-Z]+Rik\d*)$")            # mouse-only naming (predicted genes, RIKEN)
_RAT_SYM = re.compile(r"^(LOC\d+|RGD\d+|AABR\d+\.\d+|A[CY]\d+\.\d+)$")   # rat-only naming (LOC/RGD ids, contigs)


def source_feature_names(path, fmt, max_rows=200000):
    """Feature (gene) identifiers of a source matrix as two lists (ids, names), read the cheap way.

    mtx: the companion features/genes file found the way Stage 2 finds it; h5: the 10x features group;
    h5ad: the var index; csv/tsv: the first column and the header line.
    """
    import h5py
    path = Path(path)
    ids, names = [], []
    if fmt == "mtx":
        stem, prefix = path.name, None
        for suffix in (".matrix.mtx.gz", ".matrix.mtx", "_matrix.mtx.gz", "_matrix.mtx"):
            if stem.lower().endswith(suffix):
                prefix = stem[:len(stem) - len(suffix)]
                break
        cands = ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"]
        feat = next((path.parent / c for c in cands if (path.parent / c).exists()), None)
        if feat is None and prefix:
            for c in cands:
                hit = [f for f in sorted(path.parent.iterdir()) if f.is_file()
                       and f.name.lower().startswith(prefix.lower()) and f.name.lower().endswith(c)]
                if hit:
                    feat = hit[0]
                    break
        if feat is None:
            return ids, names
        opener = gzip.open if feat.name.endswith(".gz") else open
        with opener(feat, "rt", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                p = line.rstrip("\n").split("\t")
                ids.append(p[0]); names.append(p[1] if len(p) > 1 else p[0])
    elif fmt == "h5":
        dec = lambda a: [x.decode() if isinstance(x, bytes) else str(x) for x in a]   # noqa: E731
        with h5py.File(path, "r") as h:
            if "matrix" in h and "features" in h["matrix"]:
                ids = dec(h["matrix/features/id"][:]); names = dec(h["matrix/features/name"][:])
            else:
                for g in h:
                    if isinstance(h[g], h5py.Group) and "genes" in h[g]:
                        ids += dec(h[g]["genes"][:])
                        names += dec(h[g]["gene_names"][:]) if "gene_names" in h[g] else dec(h[g]["genes"][:])
    elif fmt == "h5ad":
        with h5py.File(path, "r") as h:
            v = h["var"]
            key = v.attrs.get("_index", "_index")
            key = key.decode() if isinstance(key, bytes) else key
            ids = [x.decode() if isinstance(x, bytes) else str(x) for x in v[key][:]]
            names = list(ids)
    else:
        sep = "\t" if (".tsv" in path.name or ".txt" in path.name) else ","
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", errors="replace") as f:
            header = f.readline().rstrip("\n").split(sep)
            first = []
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                first.append(line.split(sep, 1)[0].strip('"'))
        header = [x.strip('"') for x in header[1:]]
        # the gene axis is the one whose labels do not carry a cell barcode (a run of 12 or more of A, C, G, T)
        bc = re.compile(r"[ACGTN]{12,}")
        share = lambda xs: sum(bool(bc.search(x)) for x in xs[:2000]) / max(1, len(xs[:2000]))   # noqa: E731
        ids = header if share(first) > 0.5 and share(header) <= 0.5 else first
        names = list(ids)
    return ids, names


def feature_species(ids, names):
    """Classify the annotation of a feature list. Returns (label, counts).

    Ensembl prefixes decide when present (ENSRNOG rat, ENSMUSG mouse, ENSG human); otherwise symbol style:
    all-capital symbols are human; among capitalised symbols, mouse-only names (Gm1234, 1700001A01Rik) against
    rat-only names (LOC1234, RGD1234, AABR07...). 'mixed' when two genomes each hold over 5% of the features.
    """
    n = max(1, len(ids))
    c = Counter()
    for x in ids:
        x = str(x)
        core = x.split("_")[-1] if "ENS" in x else x
        if core.startswith("ENSRNOG"):
            c["ens_rat"] += 1
        elif core.startswith("ENSMUSG"):
            c["ens_mouse"] += 1
        elif core.startswith("ENSG0"):
            c["ens_human"] += 1
    for x in names:
        x = str(x).split("_")[-1] if "___" in str(x) else str(x)
        if _MOUSE_SYM.match(x):
            c["sym_mouse_only"] += 1
        elif _RAT_SYM.match(x):
            c["sym_rat_only"] += 1
        if len(x) > 2 and x.isupper() and not x.startswith(("ENS", "LOC", "RGD", "AABR", "AC", "AY")):
            c["sym_all_caps"] += 1
    c["n_features"] = len(ids)
    ens = {k: c[k] / n for k in ("ens_rat", "ens_mouse", "ens_human")}
    if sum(ens.values()) >= 0.5:
        big = [k for k, v in ens.items() if v > 0.05]
        label = "mixed" if len(big) > 1 else {"ens_rat": "rat", "ens_mouse": "mouse", "ens_human": "human"}[big[0]]
    else:
        caps, mo, ra = c["sym_all_caps"] / n, c["sym_mouse_only"] / n, c["sym_rat_only"] / n
        if caps > 0.5:
            label = "human" if (mo < 0.02 and ra < 0.02) else "mixed"
        elif mo > 0.02 and ra > 0.02:
            label = "mixed"
        elif mo > 0.02:
            label = "mouse"
        elif ra > 0.02:
            label = "rat"
        else:
            label = "unknown"
    return label, dict(c)


# ---------------------------------------------------------------------------------------------------------
# Tissue groups for the per-tissue loss table. GEO's free-text tissue strings ("Frozen postmortem nucleus
# accumbens", "Cardiac", "L4-L6 DRG") are grouped by keyword, first rule that matches. The same rule is applied
# to the held-out samples and to the training samples of the probe, so the two per-tissue tables line up.
# ---------------------------------------------------------------------------------------------------------
TISSUE_GROUPS = [
    ("brain: nucleus accumbens / striatum", r"accumbens|striat|\bnacc?\b"),
    ("brain: ventral tegmental area", r"ventral tegmental|\bvta\b"),
    ("brain: cortex / hippocampus", r"cortex|cortic|hippocamp|dentate"),
    ("brain: hypothalamus", r"hypothalam"),
    ("brain: other or unspecified", r"brain|coeruleus|amygdal|cerebell|midbrain|thalam|substantia|pallid|\bcns\b|pituitar"),
    ("brain borders (meninges, choroid plexus, CSF)", r"dura|\bpia\b|choroid|\bcsf\b|cerebrospinal|mening"),
    ("eye", r"retina|optic|\beye\b|cornea"),
    ("spinal cord", r"spinal cord"),
    ("dorsal root ganglion", r"dorsal root|\bdrg\b"),
    ("peripheral nerve", r"sciatic|nerve"),
    ("heart", r"heart|cardiac|cardio|ventric|atri(um|al)|myocard"),
    ("blood vessel", r"arter|aort|\bvein|vena cava|vascul"),
    ("lung", r"lung|pulmon|trachea|bronch|alveol"),
    ("liver", r"liver|hepat"),
    ("kidney", r"kidney|renal"),
    ("skeletal muscle", r"muscle|gastrocnem|soleus|tibialis|vastus"),
    ("adipose", r"adipos|\bfat\b|\bwat\b|\bbat\b"),
    ("skin", r"skin|derm|wound"),
    ("testis", r"testis|testes|sperm"),
    ("ovary / uterus / placenta", r"ovar|uter|placent|endometr"),
    ("spleen", r"spleen"),
    ("intestine", r"intestin|colon|ileum|jejun|duoden|\bgut\b"),
    ("blood / bone marrow", r"blood|pbmc|bone marrow|plasma"),
    ("tooth", r"molar|tooth|teeth|dental|incisor|\bpulp\b|periodont"),
    ("joint / bone / cartilage", r"menisc|cartilage|\bbone\b|joint|tendon|synov|femur|tibia|disc"),
    ("mammary gland / tumour", r"mammary|breast|tumou?r"),
    ("pancreas", r"pancrea|islet"),
]
_TISSUE_RE = [(name, re.compile(pat)) for name, pat in TISSUE_GROUPS]


def tissue_group(*texts):
    """Group of the first text that matches a rule; texts are tried in the order given (sample-level string
    first, study-level fallback after). A sample string that only says "brain" is refined by a later text that
    names a brain region. Returns 'other / unknown' when nothing matches."""
    generic = None
    for t in texts:
        t = (t or "").lower()
        for name, rx in _TISSUE_RE:
            if rx.search(t):
                if name == "brain: other or unspecified" and generic is None:
                    generic = name
                    break
                if generic and not name.startswith("brain"):
                    return generic
                return name
    return generic or "other / unknown"
