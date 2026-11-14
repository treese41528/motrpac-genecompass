"""checked_load.py -- load fine-tuned GeneCompass weights and refuse an incomplete load.

Every script that embeds or perturbs cells rebuilds the model and then calls load_state_dict(strict=False),
because the checkpoint and the freshly built model are allowed to differ in a few places. Before 2026-11-14 the
returned lists of missing and unexpected keys were discarded, so a model with randomly initialised tensors would
have produced embeddings without any message (audit finding 6.7).

load_state_checked() keeps strict=False and then
  * logs how many tensors were loaded and every missing or unexpected key;
  * raises if a model tensor is missing from the checkpoint, unless its name is in ALLOW_MISSING;
  * raises if a checkpoint tensor has no place in the model, unless its name is in ALLOW_UNEXPECTED;
  * verifies that every loaded tensor in the model equals the checkpoint's (verify=True).

ALLOW_MISSING and ALLOW_UNEXPECTED were set from the report of analysis/audit_remediation/check_weight_load.py on
the v2 and April weights (docs/audit/remediation/STEP1_INTEGRITY.md). An empty tuple means no exception is allowed.
"""
import logging

import torch

_log = logging.getLogger(__name__)

# Model tensors that the checkpoint is allowed not to hold, and checkpoint tensors the model is allowed not to use.
ALLOW_MISSING = ()
ALLOW_UNEXPECTED = ()


def load_state_checked(model, state, log=None, allow_missing=ALLOW_MISSING, allow_unexpected=ALLOW_UNEXPECTED,
                       raise_on_error=True, verify=True):
    """Load `state` into `model`; return a dict describing the load. Raises RuntimeError on an incomplete load."""
    log = log or _log
    res = model.load_state_dict(state, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    bad_missing = [k for k in missing if k not in allow_missing]
    bad_unexpected = [k for k in unexpected if k not in allow_unexpected]
    own = model.state_dict()
    shared = [k for k in own if k in state]
    n_diff, diff = 0, []
    if verify:
        for k in shared:
            a, b = own[k], state[k]
            if a.shape != b.shape or not torch.equal(a.detach().cpu(), b.detach().cpu().to(a.dtype)):
                n_diff += 1; diff.append(k)
    report = dict(n_model_tensors=len(own), n_checkpoint_tensors=len(state), n_loaded=len(shared),
                  missing=missing, unexpected=unexpected, n_loaded_but_different=n_diff, different=diff)
    log.info(f"weights: {len(shared)} of {len(own)} model tensors loaded from a checkpoint of {len(state)}; "
             f"missing {len(missing)}, unexpected {len(unexpected)}, loaded-but-different {n_diff}")
    for k in missing:
        log.warning(f"  model tensor not in the checkpoint (left at its initial value): {k}")
    for k in unexpected:
        log.warning(f"  checkpoint tensor not used by the model: {k}")
    if raise_on_error and (bad_missing or bad_unexpected or n_diff):
        raise RuntimeError(
            "incomplete weight load: "
            f"{len(bad_missing)} model tensor(s) missing from the checkpoint {bad_missing[:5]}, "
            f"{len(bad_unexpected)} checkpoint tensor(s) unused {bad_unexpected[:5]}, "
            f"{n_diff} loaded tensor(s) differ {diff[:5]}. "
            "Embeddings from this model would not be the fine-tuned model's. See finetune/genecompass/checked_load.py.")
    return report
