#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYTHON_BIN' was not found. Install Python 3.11+ or set PYTHON_BIN." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"ERROR: Python 3.11+ is required by the current MOABB installation path; "
        f"found {sys.version.split()[0]}"
    )
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Using existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p data/moabb artifacts logs

python - <<'PY'
mods = [
    "moabb",
    "mne",
    "mne_lsl",
    "numpy",
    "scipy",
    "sklearn",
    "pyriemann",
    "pydantic",
    "yaml",
]
for name in mods:
    __import__(name)
print("Dependency smoke test passed.")
PY

cat <<'TXT'

Environment setup complete.

Activate it with:
  source .venv/bin/activate

After Codex implements the package described in AGENTS.md, the intended first commands are:
  python -m bci.cli bootstrap --config configs/kalunga_v0.yaml
  python -m bci.cli evaluate  --config configs/kalunga_v0.yaml
  python -m bci.cli run       --config configs/kalunga_v0.yaml

Dataset acquisition is intentionally performed by `bootstrap`, not by setup.sh.
TXT
