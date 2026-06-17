#!/bin/bash
# Install omnideconv's Python methods (AutoGeneS + Scaden) into the SINGLE project env
# (motrpac-env) for the omnideconv benchmark replication -- one clone-and-go environment,
# NO conda, NO second venv. (Decision 2026-06-03; see deconvolution/OMNIDECONV_BENCHMARK_PLAN.md
# and memory feedback_single_reproducible_env.)
#
# WHY THIS FITS THE py3.12 GeneCompass ENV (and why no separate py3.9 venv is needed):
#   * Scaden's fork pins tensorflow==2.13.0 (no cp312 wheel; needs numpy<2) -- incompatible
#     with motrpac-env (py3.12 + torch + numpy 2.4). BUT modern TensorFlow (>=2.17) dropped
#     both caps: TF 2.21 requires only numpy>=1.26 and accepts protobuf 6.x. A pip dry-run
#     confirmed that adding tensorflow-cpu + tf-keras + the AutoGeneS fork to motrpac-env
#     changes only ONE existing package (h5py 3.16->3.14) and touches NOTHING in the
#     torch/transformers/scanpy/numpy stack. So we install Scaden on modern CPU TF + the
#     tf-keras (Keras-2) compat shim and set TF_USE_LEGACY_KERAS=1 so Scaden's Keras-2 code
#     runs unchanged. (--no-deps skips Scaden's stale TF==2.13/keras==2.13.1 pins.)
#   * tensorflow-cpu ON PURPOSE: no bundled CUDA libs to clash with torch's CUDA stack; the
#     paper ran each method single-CPU; Scaden's net is tiny. (~600MB vs multi-GB GPU TF.)
#   * AutoGeneS = the omnideconv FORK (scipy>=1.3), NOT the PyPI release (bogus scipy==1.3
#     exact-pin that tries to build a 6-year-old scipy from source).
#
# Reticulate binds to this env via RETICULATE_PYTHON (set in run_omnideconv.sh). We deliberately
# do NOT upgrade setuptools: motrpac-env has 81.0.0, which still ships the pkg_resources that
# scaden/__init__.py imports (newer setuptools removed it).
#
# Run on a Gilbreth login node (needs internet). Idempotent. ~5-10 min (TF download dominates).
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT
# Per-site profile: gives ENV_PY from site.env MGC_PYTHON and strips STRIP_CONDA from PATH (no
# modules needed for a pip install). See deconvolution/setup/site_env.sh + SETUP.md.
source "${PROJECT_ROOT}/deconvolution/setup/site_env.sh"
# pip target = the project venv: site.env MGC_PYTHON (-> ENV_PY above) -> active venv -> python3.
# Resolve from VIRTUAL_ENV BEFORE we unset it below.
ENV_PY="${ENV_PY:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
ENV_PY="${ENV_PY:-$(command -v python3 || true)}"
export TMPDIR="${PROJECT_ROOT}/tmp"; mkdir -p "${TMPDIR}"
# Keep conda + any active venv out of the way so pip targets ONLY the resolved ENV_PY.
# (site_env.sh already stripped STRIP_CONDA from PATH; here we also drop env pointers.)
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV

echo "target env : ${ENV_PY} ($(${ENV_PY} --version 2>&1))"
echo "started    : $(date)"

echo ">> [1/2] tensorflow-cpu + tf-keras + AutoGeneS (fork) + click/rich  (deps resolved)"
"${ENV_PY}" -m pip install \
  tensorflow-cpu tf-keras \
  "git+https://github.com/omnideconv/AutoGeneS.git" \
  click rich

echo ">> [2/2] scaden (fork) WITHOUT its stale TF/keras pins (--no-deps)"
"${ENV_PY}" -m pip install --no-deps "git+https://github.com/omnideconv/scaden.git"

echo ">> validating: GeneCompass stack intact + deconv methods import + Keras-2 train/predict ..."
TF_USE_LEGACY_KERAS=1 "${ENV_PY}" - <<'PY'
# 1) GeneCompass stack still imports and numpy was NOT downgraded
import torch, transformers, numpy as np, scanpy, anndata
print(f"  torch {torch.__version__} | transformers {transformers.__version__} | "
      f"numpy {np.__version__} | scanpy {scanpy.__version__} | anndata {anndata.__version__}")
assert np.__version__.startswith("2."), "numpy was downgraded -- ABORT"
# 2) deconv methods import
import autogenes, scaden, tensorflow as tf
gv = lambda m: getattr(m, "__version__", "?")
print(f"  autogenes {gv(autogenes)} | scaden {gv(scaden)} | tensorflow {tf.__version__}")
from scaden.model.scaden import Scaden
print("  Scaden model class import OK")
# 3) the Keras-2 (tf-keras) path Scaden relies on actually trains+predicts under TF 2.21
print("  tf.keras backing module:", tf.keras.__name__)
m = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(16,), activation="relu"),
                         tf.keras.layers.Dense(3, activation="softmax")])
m.compile(optimizer="adam", loss="categorical_crossentropy")
X = np.random.rand(32, 16).astype("float32")
Y = np.eye(3)[np.random.randint(0, 3, 32)].astype("float32")
m.fit(X, Y, epochs=1, batch_size=8, verbose=0)
print("  tf-keras legacy train+predict OK; pred shape", m.predict(X[:4], verbose=0).shape)
print("PY_VALIDATE_OK")
PY

echo ">> pinning deconv additions -> deconvolution/setup/requirements-omnideconv.txt"
"${ENV_PY}" -m pip freeze | grep -iE \
  '^(tensorflow|tensorflow-cpu|tf-keras|tf_keras|keras|tensorboard[a-z-]*|autogenes|scaden|deap|dill|cachetools|ml.dtypes|gast|astunparse|opt.einsum|flatbuffers|grpcio|absl.py|wrapt|termcolor|google.pasta|libclang|h5py)\b' \
  | sort > "${PROJECT_ROOT}/deconvolution/setup/requirements-omnideconv.txt" || true
echo "  $(wc -l < "${PROJECT_ROOT}/deconvolution/setup/requirements-omnideconv.txt") pinned lines"

echo "finished   : $(date)"
echo "OMNIDECONV_PY_INSTALL_DONE (unified env: ${ENV_PY})"
