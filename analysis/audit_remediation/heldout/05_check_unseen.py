#!/usr/bin/env python
"""05_check_unseen.py -- is any held-out evaluation cell a training cell under another accession?

A supplementary check, not in the registrations. STEP5_PREREG.md M2 and STEP4_PREREG.md V3 exclude a series by
its GEO sample accessions and its declared SuperSeries / SubSeries relations (01_series_list.py). The corpus
plan, however, holds seven pairs of series with byte-identical files whose relation the plan does not name
(audit Part 1, Stage 2), so accession metadata alone cannot show that a cell was never trained on. This script
compares the cells themselves, as the model sees them:

  exact    the cell's whole token sequence (2,048 ids, padding included) equals a training cell's
  prefix   its 16 highest-ranked tokens, in order, equal a training cell's (robust to a different gene filter
           lower down the ranking)

against every row of the v2 tokenized corpus (data/training/tokenized_corpus_v2/dataset, 6,803,206 rat cells,
read only), by 64-bit position-weighted hashes; every hash match is then confirmed on the token ids. The same
two comparisons are made inside the evaluation set (a cell present twice).

Writes record/step4_heldout/unseen_check.tsv (per held-out study) and unseen_check_matches.tsv (each match).
Usage: 05_check_unseen.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

L, K = 2048, 16
RNG = np.random.default_rng(20261114)
W = RNG.integers(1, 2**63 - 1, size=L, dtype=np.uint64) | np.uint64(1)


def hashes(ids):
    """ids: (n, 2048) int32 -> (full hash, prefix hash), uint64, wrap-around arithmetic."""
    x = ids.astype(np.uint64)
    with np.errstate(over="ignore"):
        return (x * W).sum(axis=1, dtype=np.uint64), (x[:, :K] * W[:K]).sum(axis=1, dtype=np.uint64)


def batches(path):
    """Yield (input_ids as (n, 2048) int32 view, study_id list, sample_id list, cell_index array) per record batch."""
    with pa.memory_map(str(path), "r") as src:
        try:
            reader = pa_ipc.open_stream(src)
        except pa.ArrowInvalid:
            reader = pa_ipc.open_file(src)
            reader = (reader.get_batch(i) for i in range(reader.num_record_batches))
        for b in reader:
            col = b.column(b.schema.get_field_index("input_ids"))
            ids = col.values.to_numpy(zero_copy_only=False).reshape(-1, L)
            names = b.schema.names
            yield (ids, b.column(names.index("study_id")).to_pylist(),
                   b.column(names.index("sample_id")).to_pylist() if "sample_id" in names else [""] * len(ids),
                   b.column(names.index("cell_index")).to_numpy(zero_copy_only=False) if "cell_index" in names
                   else np.zeros(len(ids), dtype=np.int64))


def main():
    from datasets import load_from_disk
    ev = load_from_disk(str(C.TOK_DIR / "eval_set"))
    e_ids = ev.data.column("input_ids").combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, L)
    e_study = np.asarray(ev.data.column("study_id").to_pylist())
    e_sample = np.asarray(ev.data.column("sample_id").to_pylist())
    e_cell = np.asarray(ev.data.column("cell_index").to_pylist())
    hf, hp = [], []
    for s in range(0, len(e_ids), 20000):
        a, b = hashes(e_ids[s:s + 20000]); hf.append(a); hp.append(b)
    hf, hp = np.concatenate(hf), np.concatenate(hp)
    by_full, by_pre = defaultdict(list), defaultdict(list)
    for i, (a, b) in enumerate(zip(hf.tolist(), hp.tolist())):
        by_full[a].append(i); by_pre[b].append(i)
    set_full, set_pre = np.unique(hf), np.unique(hp)

    matches, n_train = [], 0
    files = sorted((C.ROOT / "data/training/tokenized_corpus_v2/dataset").glob("data-*.arrow"))
    for fi, f in enumerate(files):
        for ids, study, sample, cidx in batches(f):
            n_train += len(ids)
            for s in range(0, len(ids), 20000):
                chunk = ids[s:s + 20000]
                a, b = hashes(chunk)
                for kind, h, table, width in (("exact", a, by_full, L), ("prefix", b, by_pre, K)):
                    hit = np.flatnonzero(np.isin(h, set_full if kind == "exact" else set_pre))
                    for j in hit.tolist():
                        for i in table[int(h[j])]:
                            if np.array_equal(chunk[j, :width], e_ids[i, :width]):      # confirm on the ids
                                matches.append({"kind": kind, "heldout_study": e_study[i], "heldout_sample": e_sample[i],
                                                "heldout_cell_index": int(e_cell[i]), "training_study": study[s + j],
                                                "training_sample": sample[s + j],
                                                "training_cell_index": int(cidx[s + j])})
        if fi % 25 == 0:
            print(f"{fi + 1}/{len(files)} corpus files, {n_train:,} training cells, {len(matches)} matches", flush=True)

    # inside the evaluation set
    internal = []
    for kind, table, width in (("exact", by_full, L), ("prefix", by_pre, K)):
        for h, rows in table.items():
            if len(rows) > 1:
                for i in rows[1:]:
                    if np.array_equal(e_ids[rows[0], :width], e_ids[i, :width]):
                        internal.append({"kind": kind, "study_a": e_study[rows[0]], "sample_a": e_sample[rows[0]],
                                         "cell_a": int(e_cell[rows[0]]), "study_b": e_study[i],
                                         "sample_b": e_sample[i], "cell_b": int(e_cell[i])})

    cells = Counter(e_study.tolist())
    ex = {k: Counter() for k in ("exact", "prefix")}
    seen = {k: set() for k in ("exact", "prefix")}
    partners = defaultdict(Counter)
    for m in matches:
        key = (m["heldout_sample"], m["heldout_cell_index"])
        if key not in seen[m["kind"]]:
            seen[m["kind"]].add(key); ex[m["kind"]][m["heldout_study"]] += 1
        partners[m["heldout_study"]][m["training_study"]] += 1
    inn = {k: Counter() for k in ("exact", "prefix")}
    for m in internal:
        inn[m["kind"]][m["study_b"]] += 1
    rows = [{"study": s, "eval_cells": cells[s], "cells_equal_to_a_training_cell_exact": ex["exact"][s],
             "cells_equal_to_a_training_cell_top16": ex["prefix"][s],
             "training_studies_matched": C.counter_str(list(partners[s].elements())),
             "repeats_inside_eval_set_exact": inn["exact"][s], "repeats_inside_eval_set_top16": inn["prefix"][s]}
            for s in sorted(cells)]
    rows.append({"study": "ALL", "eval_cells": len(e_ids),
                 "cells_equal_to_a_training_cell_exact": sum(ex["exact"].values()),
                 "cells_equal_to_a_training_cell_top16": sum(ex["prefix"].values()),
                 "training_studies_matched": f"training cells compared: {n_train}",
                 "repeats_inside_eval_set_exact": sum(inn["exact"].values()),
                 "repeats_inside_eval_set_top16": sum(inn["prefix"].values())})
    C.write_tsv(C.REC / "unseen_check.tsv", rows)
    C.write_tsv(C.REC / "unseen_check_matches.tsv", matches[:5000],
                ["kind", "heldout_study", "heldout_sample", "heldout_cell_index", "training_study", "training_sample",
                 "training_cell_index"])
    C.write_tsv(C.REC / "unseen_check_internal.tsv", internal[:5000],
                ["kind", "study_a", "sample_a", "cell_a", "study_b", "sample_b", "cell_b"])
    print(f"training cells compared: {n_train:,}; evaluation cells: {len(e_ids):,}; "
          f"exact matches: {sum(ex['exact'].values())}; top-{K} matches: {sum(ex['prefix'].values())}; "
          f"repeats inside the evaluation set: exact {sum(inn['exact'].values())}, top-{K} {sum(inn['prefix'].values())}")


if __name__ == "__main__":
    main()
