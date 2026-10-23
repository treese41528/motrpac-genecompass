#!/usr/bin/env python3
"""score_prereg_c1c2.py — score the retrain-v2 pre-registration's pass/fail criteria (C1, C2).

docs/RETRAIN_V2_PREREGISTRATION.md §3:
  C1a  every logged training step of both phases has finite id_loss and value_loss
  C1b  rat probe: v2's gene-ID loss and value loss are both lower than v2 init's
  C2   human and mouse probes: v2's gene-ID loss and value loss are each at most 10% above base's
Other eval_mlm.py results in --mlm-dir (April, v2 after phase 1) are reported for context, not gated.

  python finetune/genecompass/score_prereg_c1c2.py --mlm-dir data/results/retrain_v2/mlm \
      --trainer-state <phase-1 last checkpoint>/trainer_state.json \
      --trainer-state <phase-2 last checkpoint>/trainer_state.json \
      [--superseded-wandb <run>.wandb --resume-step N]

--mlm-dir holds eval_mlm.py outputs named base.json, init_v2.json and v2.json (+ any others).
--superseded-wandb is the local wandb file of an attempt that was resumed from checkpoint-N: its steps
>= N appear in no trainer_state, so they are checked here, and its steps < N must equal those of the
last --trainer-state. Exit 0 when C1a, C1b and C2 all pass.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ("base", "init_v2", "april", "v2_phase1", "v2")


def check_trainer_state(path):
    s = json.loads(Path(path).read_text())
    per = [e for e in s["log_history"] if "id_loss" in e]
    steps = np.array([e["step"] for e in per])
    vals = np.array([(e["id_loss"], e["value_loss"]) for e in per], float)
    once = len(steps) == s["global_step"] and np.array_equal(np.sort(steps), np.arange(s["global_step"]))
    bad = (~np.isfinite(vals)).sum(axis=0)
    print(f"{path}: {len(steps):,} logged steps, global step {s['global_step']:,} (each step logged exactly "
          f"once: {once}); non-finite id_loss {bad[0]}, value_loss {bad[1]}")
    return bool(once and not bad.any()), dict(zip(steps.tolist(), map(tuple, vals.tolist())))


def check_superseded(path, resume_step, final):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    rows, n_rec = {}, 0
    while True:
        d = ds.scan_data()
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") != "history":
            continue
        h = {i.key or "/".join(i.nested_key): i.value_json for i in r.history.item}
        if "train/id_loss" in h:
            n_rec += 1
            rows[int(float(h["train/global_step"]))] = (float(h["train/id_loss"]), float(h["train/value_loss"]))
    steps = np.array(sorted(rows))
    vals = np.array([rows[k] for k in steps], float)
    sup = steps >= resume_step
    bad = int((~np.isfinite(vals)).sum())
    same = sum(rows[k] == final.get(k) for k in steps[~sup].tolist())
    print(f"{path}: {n_rec:,} logged steps, {steps.min()}..{steps.max()}; {int(sup.sum()):,} superseded "
          f"(>= {resume_step:,}); non-finite values {bad}; steps < {resume_step:,} equal to the final run's: "
          f"{same:,}/{int((~sup).sum()):,}")
    return bad == 0 and n_rec == len(rows) and same == int((~sup).sum())


def score_mlm(mlm_dir):
    r = {p.stem: json.loads(p.read_text()) for p in sorted(Path(mlm_dir).glob("*.json"))}
    r = {k: v for k, v in r.items() if "species" in v}
    for need in ("base", "init_v2", "v2"):
        if need not in r:
            sys.exit(f"{mlm_dir}/{need}.json missing")
    names = [n for n in ORDER if n in r] + sorted(set(r) - set(ORDER))
    same = True
    for sp in ("human", "mouse", "rat"):
        seen = {(r[n]["species"][sp]["cells"], r[n]["species"][sp]["masked_tokens"])
                for n in names if sp in r[n]["species"]}
        same &= len(seen) == 1
        print(f"{sp}: (cells, masked tokens) over models = {sorted(seen)}")
    print(f"same cells and masks for every model: {same}; emb_warmup alpha "
          f"{ {n: r[n]['emb_warmup_alpha'] for n in names} }; missing keys { {n: len(r[n]['missing_keys']) for n in names} }")
    print(f"\n{'model':<10}{'species':<8}{'id_loss':>10}{'value_loss':>12}{'top1':>8}{'id vs base':>12}{'val vs base':>13}")
    base = r["base"]["species"]
    for n in names:
        for sp, m in r[n]["species"].items():
            rel = (f"{m['id_loss'] / base[sp]['id_loss'] - 1:+12.2%}{m['value_loss'] / base[sp]['value_loss'] - 1:+13.2%}"
                   if sp in base else "")
            print(f"{n:<10}{sp:<8}{m['id_loss']:>10.4f}{m['value_loss']:>12.3e}{m['top1_acc']:>8.4f}{rel}")
    v2, init = r["v2"]["species"], r["init_v2"]["species"]
    c1b = {k: v2["rat"][k] < init["rat"][k] for k in ("id_loss", "value_loss")}
    c2 = {(sp, k): v2[sp][k] <= 1.10 * base[sp][k] for sp in ("human", "mouse") for k in ("id_loss", "value_loss")}
    print()
    for k, ok in c1b.items():
        print(f"C1b rat {k}: v2 {v2['rat'][k]:.4g} vs v2 init {init['rat'][k]:.4g} -> {'pass' if ok else 'FAIL'}")
    for (sp, k), ok in c2.items():
        print(f"C2 {sp} {k}: v2 {v2[sp][k] / base[sp][k] - 1:+.2%} vs base (limit +10%) -> {'pass' if ok else 'FAIL'}")
    return same and all(c1b.values()), same and all(c2.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mlm-dir", required=True, help="eval_mlm.py outputs (base.json, init_v2.json, v2.json, ...)")
    ap.add_argument("--trainer-state", action="append", required=True,
                    help="trainer_state.json of each phase's last checkpoint, in phase order")
    ap.add_argument("--superseded-wandb", help="local .wandb file of an attempt resumed from checkpoint-N")
    ap.add_argument("--resume-step", type=int, help="N, the checkpoint step the attempt was resumed from")
    a = ap.parse_args()
    if a.superseded_wandb and a.resume_step is None:
        ap.error("--superseded-wandb needs --resume-step")
    print("== C1a: training logs")
    ok_a, final = True, {}
    for ts in a.trainer_state:
        ok, final = check_trainer_state(ts)
        ok_a &= ok
    if a.superseded_wandb:
        ok_a &= check_superseded(a.superseded_wandb, a.resume_step, final)
    print("\n== C1b, C2: fixed probe")
    ok_b, ok_2 = score_mlm(a.mlm_dir)
    verdict = lambda ok: "PASS" if ok else "FAIL"
    print(f"\nC1a {verdict(ok_a)} | C1b {verdict(ok_b)} | C2 {verdict(ok_2)}")
    sys.exit(0 if ok_a and ok_b and ok_2 else 1)


if __name__ == "__main__":
    main()
