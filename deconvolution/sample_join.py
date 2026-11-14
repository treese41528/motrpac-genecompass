"""sample_join.py -- the one place where a deconvolved sample name is turned into a MoTrPAC vial label.

Stage 8 names the bulk samples of a tissue mix1 ... mixN in the order of <bulk_root>/<TISSUE>/bulk_samples.tsv
and writes its fractions and Z matrices under those names. Every later stage needs the vial label behind a
mix name to look up sex and training group. Until the Step 3 rerun each script did that join itself, by
position, with no check, and an index outside the list became an empty label so the pseudo-cell was dropped
without a message (audit finding 6.9).

This module is the helper registered in docs/audit/remediation/STEP3_PREREG.md section 2.5:

  load_viallabels()  reads the vial labels in mix order and checks them against what Stage 8 wrote;
  mix_index()        maps mix names to positions in that list and raises on a name it cannot place.

Both raise SampleJoinError. Neither changes the join itself, which Step 1 verified in all 16 tissues
(docs/audit/remediation/STEP1_INTEGRITY.md).
"""
import csv
import re
from pathlib import Path

import numpy as np

_MIX = re.compile(r"mix(\d+)")


class SampleJoinError(ValueError):
    """The sample key of a tissue does not agree with the Stage 8 output, or a sample name has no place in it."""


def _read_lines(path):
    """Non-empty lines of a text file, stripped."""
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def load_viallabels(bulk_root, tissue, results_root=None):
    """Vial labels of one tissue in mix<i> order: element i-1 is the vial label of sample mix<i>.

    bulk_root     folder holding <TISSUE>/bulk_samples.tsv (one vial label per line, no header).
    tissue        tissue name; the folder name is its upper-case form.
    results_root  optional folder holding the Stage 8 output of each tissue, <TISSUE>/ (the config's
                  deconvolution.results_dir + "/motrpac"). None, or a tree without the tissue, skips the checks,
                  so a script still runs where the Stage 8 output is not on disk.

    Checks (section 2.5), in this order:
      1. <results_root>/<TISSUE>/mixture_samples.tsv exists (tab-separated, header "mix<TAB>viallabel"; Stage 8
         writes it from the rerun on): its rows must be mix1 ... mixN with exactly these vial labels in this
         order.
      2. otherwise, <results_root>/<TISSUE>/estimated_fractions.csv exists: its number of sample rows must
         equal the number of vial labels.
    """
    tissue = str(tissue).upper()
    samples_file = Path(bulk_root) / tissue / "bulk_samples.tsv"
    vials = _read_lines(samples_file)
    if not vials:
        raise SampleJoinError(f"[{tissue}] no vial labels in {samples_file}")
    if results_root is None:
        return vials

    tissue_dir = Path(results_root) / tissue
    key_file = tissue_dir / "mixture_samples.tsv"
    fractions_file = tissue_dir / "estimated_fractions.csv"
    if key_file.exists():
        with open(key_file, newline="") as fh:
            rows = [row for row in csv.reader(fh, delimiter="\t") if row and any(cell.strip() for cell in row)]
        header = [cell.strip() for cell in rows[0]] if rows else []
        if header != ["mix", "viallabel"]:
            raise SampleJoinError(f"[{tissue}] {key_file} must have the header 'mix<TAB>viallabel', has {header}")
        key = [(row[0].strip(), row[1].strip()) for row in rows[1:]]
        want = [(f"mix{i + 1}", vial) for i, vial in enumerate(vials)]
        if key != want:
            if len(key) != len(want):
                detail = f"{len(key)} rows against {len(want)} vial labels"
            else:
                first = next(i for i, (a, b) in enumerate(zip(key, want)) if a != b)
                detail = f"first difference at row {first + 1}: key file {key[first]}, bulk_samples.tsv {want[first]}"
            raise SampleJoinError(
                f"[{tissue}] the Stage 8 sample key {key_file} does not match {samples_file} ({detail}). "
                "The pseudo-cells of this tissue cannot be joined to animals by position.")
    elif fractions_file.exists():
        with open(fractions_file, newline="") as fh:
            n_rows = sum(1 for row in csv.reader(fh) if row) - 1          # minus the header row
        if n_rows != len(vials):
            raise SampleJoinError(
                f"[{tissue}] {fractions_file} has {n_rows} samples and {samples_file} has {len(vials)} vial "
                "labels. The pseudo-cells of this tissue cannot be joined to animals by position.")
    return vials


def mix_index(samples, n_samples, context=""):
    """Zero-based positions of mix<i> sample names in a list of n_samples vial labels.

    A name that is not 'mix' followed by digits, or whose number is outside 1 ... n_samples, raises. Before the
    Step 3 rerun such a sample got an empty vial label and its pseudo-cell left the analysis silently
    (section 2.5: "a sample index outside the list is an error where it used to be dropped silently").
    """
    names = [str(s) for s in samples]
    index = np.empty(len(names), dtype=np.int64)
    bad = []
    for k, name in enumerate(names):
        match = _MIX.fullmatch(name)
        i = int(match.group(1)) if match else 0
        if not match or not 1 <= i <= n_samples:
            bad.append(name)
        index[k] = i - 1
    if bad:
        where = f"[{context}] " if context else ""
        shown = ", ".join(sorted(set(bad))[:10])
        raise SampleJoinError(
            f"{where}{len(bad)} sample name(s) have no place among the {n_samples} vial labels "
            f"(expected mix1 ... mix{n_samples}): {shown}")
    return index
