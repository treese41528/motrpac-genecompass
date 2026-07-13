"""celltype_names.py -- the ONE definition of the cell-type -> filename contract (Python side).

Mirror of deconvolution/R/celltype_names.R; see that file for the full history. Short
version: until 2026-07-12 every script carried its own copy of

    def safe(s): return re.sub(r'[^A-Za-z0-9]+', '_', s)

which is lossy and NOT injective. KIDNEY's 'alpha-intercalated cells' and
'beta-intercalated cells' (written with Greek letters) both collapsed to
'_intercalated_cells', so extract_z.R destroyed one cell type's BayesPrism Z and the DE
then ran the same Z for both blocks. Import safe() from here; do not re-implement it.
"""

import os
import re

# Transliterated first so the map is injective. Byte-identical to the legacy sanitizer
# for every pure-ASCII label -- only non-ASCII labels change name.
CT_TRANSLIT = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'σ': 'sigma',
    'τ': 'tau', 'ω': 'omega',
    'ä': 'a', 'å': 'a', 'ç': 'c', 'è': 'e', 'é': 'e',
    'í': 'i', 'ñ': 'n', 'ö': 'o', 'ü': 'u',
    'Ä': 'A', 'Ö': 'O', 'Ü': 'U',
}


def ct_translit(s: str) -> str:
    for k, v in CT_TRANSLIT.items():
        s = s.replace(k, v)
    return s


def safe(s: str) -> str:
    """Cell type -> filename stem."""
    return re.sub(r'[^A-Za-z0-9]+', '_', ct_translit(str(s))).strip('_')


def legacy_safe(s: str) -> str:
    """The pre-2026-07-12 sanitizer. Only for locating artifacts written before the fix."""
    return re.sub(r'[^A-Za-z0-9]+', '_', str(s))


def assert_injective(types, context: str = '') -> bool:
    """Hard-fail on a colliding label set rather than silently overwriting a file."""
    seen: dict = {}
    for t in types:
        seen.setdefault(safe(t), []).append(t)
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup:
        detail = '\n'.join(f"    {k:<30} <- {v}" for k, v in dup.items())
        raise ValueError(
            f"cell-type name collision{' in ' + context if context else ''}: "
            f"{len(dup)} filename(s) claimed by more than one cell type.\n{detail}\n"
            "  A collision silently overwrites one cell type's data. Add the offending\n"
            "  character(s) to CT_TRANSLIT in deconvolution/celltype_names.py."
        )
    return True


def purge_stale(dirpath, types, prefix: str, ext: str) -> list:
    """Delete per-cell-type artifacts left by a previous run whose labels no longer exist.

    The other half of the bug: writers that never cleaned up left orphans behind whenever a
    tissue was relabelled (or a collided name was fixed), and any glob would happily ingest them.
    """
    import glob as _glob
    keep = {f'{prefix}{safe(t)}{ext}' for t in types}
    dead = [p for p in _glob.glob(os.path.join(str(dirpath), f'{prefix}*{ext}'))
            if os.path.basename(p) not in keep]
    for p in dead:
        os.remove(p)
    if dead:
        print(f"  purged {len(dead)} stale file(s) in {os.path.basename(str(dirpath))}: "
              f"{', '.join(sorted(os.path.basename(p) for p in dead)[:4])}")
    return dead


def resolve(dirpath, cell_type: str, prefix: str, ext: str) -> str:
    """Path to a per-cell-type artifact, tolerating files written by the legacy sanitizer.

    Prefers the current name; falls back to the legacy name only if that file exists.
    Lets consumers read pre-fix artifacts (e.g. the purity-sweep true-Z) without a rebuild.
    """
    p = os.path.join(str(dirpath), f'{prefix}{safe(cell_type)}{ext}')
    if os.path.exists(p):
        return p
    legacy = os.path.join(str(dirpath), f'{prefix}{legacy_safe(cell_type)}{ext}')
    return legacy if os.path.exists(legacy) else p
