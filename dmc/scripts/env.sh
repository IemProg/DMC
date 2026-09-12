#!/bin/bash
# ---------------------------------------------------------------------------
# Shared environment for all DMC scripts.
#
# Every script under dmc/scripts/ sources this file to resolve repository
# paths, so the repository can be cloned anywhere and run from any working
# directory. Nothing here is machine specific.
#
# Override the two locations that hold large files via environment variables:
#
#   DMC_DATA     where the datasets live      (default: <repo>/DATA)
#   DMC_OUTPUT   where runs are written       (default: <repo>/output)
#
# Example:
#   export DMC_DATA=/path/to/datasets
#   export DMC_OUTPUT=/path/to/dmc-runs
# ---------------------------------------------------------------------------

# Directory containing this file: <repo>/dmc/scripts
_DMC_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# <repo>/dmc — the Python package (train.py, trainers/, configs/, datasets/)
CODE="$(cd "${_DMC_SCRIPTS_DIR}/.." && pwd)"

# <repo> — the repository root
REPO="$(cd "${CODE}/.." && pwd)"

TRAIN_PY="${CODE}/train.py"
CONFIGS="${CODE}/configs"

DATA="${DMC_DATA:-${REPO}/DATA}"
OUTPUT="${DMC_OUTPUT:-${REPO}/output}"

# Run everything from the package directory so that `import trainers.*` and
# `import datasets.*` in train.py resolve against this repository.
cd "${CODE}"
