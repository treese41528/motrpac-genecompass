#!/usr/bin/env python
"""07_tables.py -- the loss tables of the held-out test and the registered reading.

Implements the reporting and the criterion of STEP4_PREREG.md V3:
  ratio = v2's gene-identity loss on the held-out evaluation cells / v2's gene-identity loss on the in-sample
          rat probe of C1;  "Generalises" if at most 1.10, "Partly" up to 1.25, "Does not" above.
"Reported with it: the value loss, the per-study and per-tissue losses, the number of studies and cells, and
the per-tier loss." All losses are token-weighted over masked tokens, as in eval_mlm.py.

Reads the evaluator output of 06_eval.py (data/audit_remediation/heldout/eval/<model>_<set>.json and
.cells.npz; model in v2, april; set in probe, heldout), the unchanged eval_mlm.py rerun on the probe
(eval/evalmlm_<model>_probe.json) and the C1/C2 record (data/results/retrain_v2/mlm/, mlm_by_tier/).
Stops if the v2 probe loss is not the recorded one.

Writes (record/step4_heldout/):
  probe_reproduction.tsv   the recorded C1 losses, the eval_mlm.py rerun and 06_eval.py, side by side
  losses_overall.tsv       model x set
  losses_by_study.tsv      model x set x study (the probe's rat cells by their training study)
  losses_by_tissue.tsv     model x set x tissue group (common.tissue_group, the same keyword rule for both sets)
  losses_by_tier.tsv       model x set x ortholog tier of the masked token
  losses_by_length.tsv     model x set x number of expressed genes in the cell (context, not registered)
  heldout_series.tsv       one row per candidate series: exclusions, QC counts, cells drawn, losses
  registered_reading.tsv   the ratio and the reading, with unregistered context lines marked as such
Usage: 07_tables.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C   # noqa: E402

MODELS = {"v2": "data/models/rat_genecompass_finetuned_v2/models/rat_phase2_mixed_species/models",
          "april": "data/models/rat_genecompass_finetuned/models/rat_phase2_mixed_species/models"}
RECORD = C.ROOT / "data/results/retrain_v2"
LEN_BINS = [(0, 499, "under 500"), (500, 999, "500-999"), (1000, 2047, "1000-2047"), (2048, 2048, "2048 (full)")]


def load(model, s):
    j = C.load_json(C.EVAL_DIR / f"{model}_{s}.json")
    z = np.load(C.EVAL_DIR / f"{model}_{s}.cells.npz", allow_pickle=False)
    return j, {k: z[k] for k in z.files}


def agg(z, mask):
    n = z["n_masked"][mask].sum()
    return {"cells": int(mask.sum()), "masked_tokens": int(n), "id_loss": float(z["ce_sum"][mask].sum() / n),
            "value_loss": float(z["se_sum"][mask].sum() / n), "top1_acc": float(z["n_hit"][mask].sum() / n),
            "mean_expressed_genes": float(z["length"][mask].mean())}


def main():
    # tissue per sample: the held-out plan, and the corpus inventory for the probe's training samples
    tissue = {r["sample_id"]: r["tissue_group"] for r in C.read_tsv(C.REC / "file_plan.tsv")}
    inv_tissue = {r["sample_id"]: C.tissue_group(r.get("tissue_normalized"), r.get("tissue_resolved"))
                  for r in C.read_tsv(C.INVENTORY)}
    titles = {r["accession"]: r["title"] for r in C.read_tsv(C.REC / "series_list.tsv")}

    # ---- the probe is the registered one, and its recorded loss is reproduced -----------------------------
    rep = []
    for model in MODELS:
        rec = C.load_json(RECORD / "mlm" / f"{model}.json")["species"]["rat"]
        rec_t = C.load_json(RECORD / "mlm_by_tier" / f"{model}.json")
        rep.append({"model": model, "source": f"record: data/results/retrain_v2/mlm/{model}.json",
                    "model_dir": C.load_json(RECORD / "mlm" / f"{model}.json")["model"], **{k: rec[k] for k in
                    ("cells", "masked_tokens", "id_loss", "value_loss", "top1_acc")}})
        p = C.EVAL_DIR / f"evalmlm_{model}_probe.json"
        if p.is_file():
            e = C.load_json(p)
            rep.append({"model": model, "source": "rerun: finetune/genecompass/eval_mlm.py eval --by-tier (unchanged)",
                        "model_dir": e["model"], **{k: e["species"]["rat"][k] for k in
                        ("cells", "masked_tokens", "id_loss", "value_loss", "top1_acc")}})
            for sp in ("human", "mouse"):
                r0 = C.load_json(RECORD / "mlm" / f"{model}.json")["species"][sp]
                rep.append({"model": model, "source": f"rerun, {sp} probe cells (record id_loss {r0['id_loss']}, "
                                                      f"value_loss {r0['value_loss']})",
                            "model_dir": e["model"], **{k: e["species"][sp][k] for k in
                            ("cells", "masked_tokens", "id_loss", "value_loss", "top1_acc")}})
            bt = e["species"]["rat"].get("by_tier", {})
            same_t = all(abs(bt[t]["id_loss"] - rec_t["species"]["rat"]["by_tier"][t]["id_loss"]) < 1e-6 * bt[t]["id_loss"]
                         for t in bt) and set(bt) == set(rec_t["species"]["rat"]["by_tier"])
            rep[-3]["per_tier_equals_record"] = same_t
        j, _ = load(model, "probe")
        mine = j["species"]["rat"]
        rep.append({"model": model, "source": "06_eval.py (eval_mlm.py's functions, per cell)", "model_dir": j["model"],
                    **{k: mine[k] for k in ("cells", "masked_tokens", "id_loss", "value_loss", "top1_acc")}})
        for r in rep:
            if r["model"] == model and not r["source"].startswith("rerun, "):
                r["id_loss_rel_diff_from_record"] = (r["id_loss"] - rec["id_loss"]) / rec["id_loss"]
                r["value_loss_rel_diff_from_record"] = (r["value_loss"] - rec["value_loss"]) / rec["value_loss"]
    C.write_tsv(C.REC / "probe_reproduction.tsv", rep,
                ["model", "source", "model_dir", "cells", "masked_tokens", "id_loss", "value_loss", "top1_acc",
                 "id_loss_rel_diff_from_record", "value_loss_rel_diff_from_record", "per_tier_equals_record"])
    bad = [r for r in rep if r["model"] == "v2" and "id_loss_rel_diff_from_record" in r
           and (abs(r["id_loss_rel_diff_from_record"]) > 1e-6 or r["masked_tokens"] != rep[0]["masked_tokens"])]
    if bad:
        sys.exit(f"STOP: the v2 in-sample probe loss is not the recorded one: {json.dumps(bad, indent=1)}")

    overall, by_study, by_tissue, by_tier, by_len, keep = [], [], [], [], [], {}
    for model in MODELS:
        for s in ("probe", "heldout"):
            j, z = load(model, s)
            rat = j["species"]["rat"]
            chk = agg(z, np.ones(len(z["row"]), dtype=bool))
            assert chk["masked_tokens"] == rat["masked_tokens"] and abs(chk["id_loss"] - rat["id_loss"]) < 1e-5 * rat["id_loss"]
            keep[(model, s)] = (rat, z)
            studies = sorted(set(z["study_id"].tolist()))
            overall.append({"model": model, "set": s, "model_dir": j["model"], "eval_set": j["probe"],
                            "studies": len(studies), "cells": rat["cells"], "masked_tokens": rat["masked_tokens"],
                            "id_loss": rat["id_loss"], "value_loss": rat["value_loss"], "top1_acc": rat["top1_acc"],
                            "mean_expressed_genes": chk["mean_expressed_genes"], "seed": j["seed"],
                            "batch_size": j["batch_size"], "device": j.get("device", "")})
            tmap = tissue if s == "heldout" else inv_tissue
            tis = np.asarray([tmap.get(x, "other / unknown") for x in z["sample_id"].tolist()])
            for st in studies:
                m = z["study_id"] == st
                by_study.append({"model": model, "set": s, "study": st, "title": titles.get(st, ""),
                                 "tissues": C.counter_str(tis[m].tolist()), **agg(z, m)})
            for t in sorted(set(tis.tolist())):
                m = tis == t
                by_tissue.append({"model": model, "set": s, "tissue_group": t,
                                  "studies": len(set(z["study_id"][m].tolist())), **agg(z, m)})
            for i, t in enumerate(z["tier_names"].tolist()):
                n = z["tier_n"][:, i].sum()
                if n > 0:
                    by_tier.append({"model": model, "set": s, "tier": t, "masked_tokens": int(n),
                                    "share_of_masked_tokens": float(n / z["n_masked"].sum()),
                                    "id_loss": float(z["tier_ce"][:, i].sum() / n),
                                    "top1_acc": float(z["tier_hit"][:, i].sum() / n)})
            for lo, hi, name in LEN_BINS:
                m = (z["length"] >= lo) & (z["length"] <= hi)
                if m.any():
                    by_len.append({"model": model, "set": s, "expressed_genes": name,
                                   "share_of_cells": float(m.mean()), **agg(z, m)})
    for r in by_study:
        r["id_loss_over_probe"] = r["id_loss"] / keep[(r["model"], "probe")][0]["id_loss"]
    # one row per candidate series: what happened to it at every step, and its losses
    qc = {r["series"]: r for r in C.read_tsv(C.REC / "qc_by_series.tsv")}
    draw = {r["study"]: r for r in C.read_tsv(C.REC / "eval_draw.tsv")}
    loss = {(r["model"], r["set"], r["study"]): r for r in by_study}
    summary = []
    for r in C.read_tsv(C.REC / "series_list.tsv"):
        if r["status"] != "candidate":
            continue
        a = r["accession"]
        q, d = qc.get(a, {}), draw.get(a, {})
        v, ap = loss.get(("v2", "heldout", a), {}), loss.get(("april", "heldout", a), {})
        summary.append({
            "series": a, "title": r["title"], "llm_tissues": r["llm_tissues"],
            "tissue_groups": q.get("tissue_groups_kept", ""), "n_gsm": r["n_gsm_soft"],
            "series_excluded": r["series_excluded"], "series_exclusion_reason": r["series_exclusion_reason"],
            "series_note": r["series_note"],
            "files_discovered": q.get("files_discovered", ""), "files_qc_success": q.get("files_qc_success", ""),
            "cells_passing_qc": q.get("cells_passing_qc", ""), "files_kept": q.get("files_kept", ""),
            "cells_kept": q.get("cells_kept", ""), "cells_drop_nonrat": q.get("cells_drop_nonrat", ""),
            "cells_drop_not_raw_counts": q.get("cells_drop_not_raw_counts", ""),
            "cells_drop_duplicate": q.get("cells_drop_duplicate", ""),
            "cells_kept_spatial_assay": q.get("cells_kept_spatial_assay", ""),
            "cells_drawn": d.get("cells_drawn", 0),
            "v2_id_loss": v.get("id_loss", ""), "v2_value_loss": v.get("value_loss", ""),
            "v2_top1_acc": v.get("top1_acc", ""), "v2_id_loss_over_probe": v.get("id_loss_over_probe", ""),
            "april_id_loss": ap.get("id_loss", ""), "april_value_loss": ap.get("value_loss", ""),
            "mean_expressed_genes": v.get("mean_expressed_genes", ""),
        })
    C.write_tsv(C.REC / "heldout_series.tsv", summary)
    C.write_tsv(C.REC / "losses_overall.tsv", overall)
    C.write_tsv(C.REC / "losses_by_study.tsv", by_study)
    C.write_tsv(C.REC / "losses_by_tissue.tsv", by_tissue)
    C.write_tsv(C.REC / "losses_by_tier.tsv", by_tier)
    C.write_tsv(C.REC / "losses_by_length.tsv", by_len)

    # ---- the registered reading ---------------------------------------------------------------------------
    (p, zp), (h, zh) = keep[("v2", "probe")], keep[("v2", "heldout")]
    ratio = h["id_loss"] / p["id_loss"]
    reading = "Generalises" if ratio <= 1.10 else "Partly" if ratio <= 1.25 else "Does not"
    rows = [("v2_probe_id_loss", p["id_loss"], "registered"), ("v2_heldout_id_loss", h["id_loss"], "registered"),
            ("ratio_v2_heldout_over_v2_probe_id_loss", ratio, "registered"),
            ("registered_reading", reading, "registered: Generalises <= 1.10; Partly <= 1.25; Does not above"),
            ("heldout_studies", len(set(zh["study_id"].tolist())), "reported with it"),
            ("heldout_cells", h["cells"], "reported with it"),
            ("heldout_masked_tokens", h["masked_tokens"], "reported with it"),
            ("probe_cells", p["cells"], "reported with it"),
            ("probe_studies", len(set(zp["study_id"].tolist())), "reported with it"),
            ("v2_probe_value_loss", p["value_loss"], "reported with it"),
            ("v2_heldout_value_loss", h["value_loss"], "reported with it"),
            ("ratio_v2_value_loss", h["value_loss"] / p["value_loss"], "reported with it (no registered threshold)")]
    (pa_, _), (ha_, _) = keep[("april", "probe")], keep[("april", "heldout")]
    rows += [("april_probe_id_loss", pa_["id_loss"], "comparison (April was trained on other values; context)"),
             ("april_heldout_id_loss", ha_["id_loss"], "comparison"),
             ("ratio_april_heldout_over_april_probe_id_loss", ha_["id_loss"] / pa_["id_loss"], "comparison"),
             ("april_probe_value_loss", pa_["value_loss"], "comparison"),
             ("april_heldout_value_loss", ha_["value_loss"], "comparison")]

    # unregistered context: how much the ratio depends on which studies are in the set
    st = sorted(set(zh["study_id"].tolist()))
    ce = np.array([zh["ce_sum"][zh["study_id"] == s].sum() for s in st])
    nn = np.array([zh["n_masked"][zh["study_id"] == s].sum() for s in st])
    per = ce / nn / p["id_loss"]
    rows += [("context_per_study_ratio_median", float(np.median(per)), "not registered"),
             ("context_per_study_ratio_min", f"{per.min():.6f} ({st[int(per.argmin())]})", "not registered"),
             ("context_per_study_ratio_max", f"{per.max():.6f} ({st[int(per.argmax())]})", "not registered"),
             ("context_studies_at_or_under_1.10", int((per <= 1.10).sum()), "not registered"),
             ("context_studies_1.10_to_1.25", int(((per > 1.10) & (per <= 1.25)).sum()), "not registered"),
             ("context_studies_above_1.25", int((per > 1.25).sum()), "not registered")]
    loo = np.array([(ce.sum() - ce[i]) / (nn.sum() - nn[i]) for i in range(len(st))]) / p["id_loss"]
    rows += [("context_leave_one_study_out_ratio_range", f"{loo.min():.6f} to {loo.max():.6f}", "not registered")]
    rng = np.random.default_rng(C.DRAW_SEED)
    b = rng.integers(0, len(st), size=(10000, len(st)))
    boot = ce[b].sum(1) / nn[b].sum(1) / p["id_loss"]
    rows += [("context_study_bootstrap_95pct_interval_of_ratio",
              f"{np.quantile(boot, 0.025):.6f} to {np.quantile(boot, 0.975):.6f}",
              "not registered: 10,000 resamples of the held-out studies, probe loss fixed")]
    # the in-sample spread across training studies, for scale
    pst = sorted(set(zp["study_id"].tolist()))
    pper = np.array([zp["ce_sum"][zp["study_id"] == s].sum() / zp["n_masked"][zp["study_id"] == s].sum()
                     for s in pst if (zp["study_id"] == s).sum() >= 20]) / p["id_loss"]
    rows += [("context_probe_per_study_ratio_range_studies_with_20_cells",
              f"{pper.min():.6f} to {pper.max():.6f} (median {np.median(pper):.6f}, {len(pper)} studies)",
              "not registered: in-sample studies' loss over the pooled probe loss")]
    notes = {r["accession"]: r["rule"] for r in C.read_tsv(C.REC / "series_exclusions.tsv") if r["kind"] == "note"}
    sib = [s for s in st if notes.get(s) == "sibling_of_training_under_superseries"]
    if sib:
        m = ~np.isin(st, sib)
        rows += [(f"context_ratio_without_{'_'.join(sib)}", float(ce[m].sum() / nn[m].sum() / p["id_loss"]),
                  "not registered: without the series that shares a SuperSeries with a training series")]
    spatial = {r["sample_id"] for r in C.read_tsv(C.REC / "file_plan.tsv") if r.get("spatial_assay") == "True"}
    sp = np.isin(zh["sample_id"], sorted(spatial)) if spatial else np.zeros(len(zh["sample_id"]), dtype=bool)
    if sp.any():
        rows += [("context_spatial_assay_cells_in_evaluation_set", int(sp.sum()),
                  "not registered: cells of files whose GEO sample names a spatial assay (spots, not cells)"),
                 ("context_spatial_assay_id_loss", float(zh["ce_sum"][sp].sum() / zh["n_masked"][sp].sum()), "not registered"),
                 ("context_ratio_without_spatial_assay_cells",
                  float(zh["ce_sum"][~sp].sum() / zh["n_masked"][~sp].sum() / p["id_loss"]), "not registered")]
    C.write_tsv(C.REC / "registered_reading.tsv", [{"quantity": k, "value": v, "status": s} for k, v, s in rows])
    for k, v, s in rows:
        print(f"{k}\t{v}\t{s}")


if __name__ == "__main__":
    main()
