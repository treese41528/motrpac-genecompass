#!/usr/bin/env python3
"""
Verify raw integer counts in ALL QC'd h5ad files.
Checks .raw.X for integer values, flags any anomalies.

Usage: python pipeline/02_gene_universe/verify_raw_counts.py [--qc-dir DIR] [--out-dir DIR]
SLURM: see verify_raw_counts_all.slurm (local)
"""

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime

import h5py
import numpy as np

QC_DIR = "/depot/reese18/data/training/preprocessed/qc_matrices"
OUT_DIR = "/depot/reese18/data/training/preprocessed/raw_count_verification"


def check_file(fpath):
    """Check one h5ad's raw/X. Returns (record dict, list of issue strings)."""
    fname = os.path.basename(fpath)
    rec = {"file": fname, "study": fname.split("_sample")[0]}
    issues = []
    try:
        with h5py.File(fpath, 'r') as h:
            # --- Check .raw exists ---
            if 'raw' not in h:
                rec["status"] = "NO_RAW"
                return rec, issues
            if 'X' not in h['raw']:
                rec["status"] = "NO_RAW_X"
                return rec, issues

            # --- Read raw/X data sample ---
            raw_x = h['raw/X']
            if isinstance(raw_x, h5py.Group):
                # Sparse matrix: sample from beginning, middle, and end
                total_nnz = raw_x['data'].shape[0]
                if total_nnz <= 1000:
                    data = raw_x['data'][:]
                else:
                    idx_start = raw_x['data'][:334]
                    idx_mid = raw_x['data'][total_nnz//2 : total_nnz//2 + 333]
                    idx_end = raw_x['data'][-333:]
                    data = np.concatenate([idx_start, idx_mid, idx_end])
                rec["storage"] = "sparse"
                rec["nnz"] = int(total_nnz)
            else:
                # Dense matrix
                flat = raw_x[:].flatten()
                data = flat[flat != 0][:1000]
                rec["storage"] = "dense"
                rec["shape"] = list(raw_x.shape)

            # --- Integer check ---
            rec["dtype"] = str(data.dtype)
            is_integer = bool(np.allclose(data, np.round(data)))
            rec["is_integer"] = is_integer
            rec["min"] = float(data.min())
            rec["max"] = float(data.max())
            rec["mean"] = float(data.mean())

            if not is_integer:
                fractional = data[~np.isclose(data, np.round(data))]
                rec["n_fractional"] = int(len(fractional))
                rec["fractional_sample"] = fractional[:10].tolist()
                rec["status"] = "NOT_INTEGER"
            else:
                rec["status"] = "OK"

            # --- Check for negative values (shouldn't exist in counts) ---
            if data.min() < 0:
                rec["has_negatives"] = True
                issues.append(f"{fname}: negative values in raw/X (min={data.min():.4f})")

            # --- Check for very large values (potential normalization artifact) ---
            if data.max() > 100000:
                rec["suspicious_max"] = True
                issues.append(f"{fname}: suspiciously large max in raw/X ({data.max():.0f})")

            # --- Gene ID format check ---
            if 'var' in h['raw'] and '_index' in h['raw/var']:
                first_gene = h['raw/var/_index'][0]
                if isinstance(first_gene, bytes):
                    first_gene = first_gene.decode()
                rec["gene_id_prefix"] = first_gene[:15]

            # --- Cell count ---
            if 'obs' in h and '_index' in h['obs']:
                rec["n_cells"] = int(h['obs/_index'].shape[0])

    except Exception as e:
        issues.append(f"{fname}: ERROR - {str(e)}")
        rec["status"] = "ERROR"
        rec["error"] = str(e)
    return rec, issues


def gene_id_format(prefix):
    if prefix.startswith("ENSRNOG"):
        return "ENSRNOG"
    if prefix.startswith("ENS"):
        return f"other_ENS:{prefix}"
    return f"non_ensembl:{prefix}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc-dir", default=QC_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.qc_dir, "*.h5ad")))
    print(f"Total QC'd h5ad files: {len(files)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results, issues = [], []
    for i, fpath in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(files)} ({100*(i+1)/len(files):.1f}%)")
        rec, file_issues = check_file(fpath)
        results.append(rec)
        issues.extend(file_issues)

    dtype_counts = Counter(r["dtype"] for r in results if "dtype" in r)
    gene_id_formats = Counter(gene_id_format(r["gene_id_prefix"]) for r in results if "gene_id_prefix" in r)
    not_integer_files = [r["file"] for r in results if r.get("status") == "NOT_INTEGER"]
    no_raw_files = [r["file"] for r in results if r.get("status") == "NO_RAW"]

    # === SUMMARY ===
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_not_int = len(not_integer_files)
    n_no_raw = len(no_raw_files)
    n_no_raw_x = sum(1 for r in results if r.get("status") == "NO_RAW_X")
    n_error = sum(1 for r in results if r.get("status") == "ERROR")

    print(f"\nTotal files:          {len(files)}")
    print(f"  OK (integer counts): {n_ok}  ({'✓' if n_ok == len(files) else '⚠'}")
    print(f"  NOT integer:         {n_not_int}")
    print(f"  No .raw:             {n_no_raw}")
    print(f"  No .raw.X:           {n_no_raw_x}")
    print(f"  Errors:              {n_error}")

    print(f"\nDtype distribution:")
    for dtype, count in dtype_counts.most_common():
        print(f"  {dtype}: {count}")

    print(f"\nGene ID format:")
    for fmt, count in gene_id_formats.most_common():
        print(f"  {fmt}: {count}")

    if not_integer_files:
        print(f"\n⚠ FILES WITH NON-INTEGER raw/X:")
        for f in not_integer_files:
            rec = next(r for r in results if r["file"] == f)
            print(f"  {f}: dtype={rec['dtype']}, sample={rec.get('fractional_sample', [])[:5]}")

    if no_raw_files:
        print(f"\n⚠ FILES MISSING .raw:")
        for f in no_raw_files:
            print(f"  {f}")

    if issues:
        print(f"\n⚠ OTHER ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  {issue}")

    if n_ok == len(files):
        print(f"\n✓ ALL {len(files)} FILES VERIFIED: raw integer counts available")
        print("  → Stage 5 GeneCompass normalization can proceed")

    # --- Compute aggregate stats ---
    ok_results = [r for r in results if r.get("status") == "OK"]
    if ok_results:
        total_cells = sum(r.get("n_cells", 0) for r in ok_results)
        total_nnz = sum(r.get("nnz", 0) for r in ok_results if r.get("storage") == "sparse")
        print(f"\nAggregate stats (OK files only):")
        print(f"  Total cells: {total_cells:,}")
        print(f"  Total non-zero entries (sparse): {total_nnz:,}")

    # === Save detailed results ===
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(files),
        "ok_count": n_ok,
        "not_integer_count": n_not_int,
        "no_raw_count": n_no_raw,
        "no_raw_x_count": n_no_raw_x,
        "error_count": n_error,
        "dtype_distribution": dict(dtype_counts),
        "gene_id_formats": dict(gene_id_formats),
        "not_integer_files": not_integer_files,
        "no_raw_files": no_raw_files,
        "issues": issues,
        "all_verified": n_ok == len(files),
    }

    with open(os.path.join(args.out_dir, "verification_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(args.out_dir, "verification_details.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    with open(os.path.join(args.out_dir, "verification_summary.txt"), 'w') as f:
        f.write(f"Raw Count Verification Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files: {len(files)}\n")
        f.write(f"OK (integer counts): {n_ok}\n")
        f.write(f"Not integer: {n_not_int}\n")
        f.write(f"No .raw: {n_no_raw}\n")
        f.write(f"Errors: {n_error}\n")
        f.write(f"All verified: {n_ok == len(files)}\n")
        if ok_results:
            f.write(f"Total cells: {total_cells:,}\n")
            f.write(f"Total nnz (sparse): {total_nnz:,}\n")

    print(f"\nResults saved to: {args.out_dir}/")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
