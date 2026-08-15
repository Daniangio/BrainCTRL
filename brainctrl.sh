#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "No .venv found; running setup.sh first."
  ./setup.sh
fi

# shellcheck disable=SC1091
if [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate     # Se rileva Windows (Git Bash)
else
    source .venv/bin/activate        # Se rileva Linux / macOS / WSL
fi

mapfile -t CONFIGS < <(find configs -maxdepth 1 -name '*.yaml' | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No YAML configs found under configs/." >&2
  exit 1
fi

echo "BrainCTRL"
echo "========="
echo
echo "Configuration:"
for i in "${!CONFIGS[@]}"; do
  printf "  %d) %s\n" "$((i + 1))" "${CONFIGS[$i]}"
done
read -r -p "Select config [1]: " CONFIG_CHOICE
CONFIG_CHOICE="${CONFIG_CHOICE:-1}"
CONFIG="${CONFIGS[$((CONFIG_CHOICE - 1))]}"

read -r -p "Subject [1]: " SUBJECT
SUBJECT="${SUBJECT:-1}"

echo
echo "Mode:"
echo "  1) Realtime replay"
echo "  2) Fast offline evaluation"
echo "  3) Bootstrap dataset only"
echo "  4) Classifier smoke tutorial"
echo "  5) Controller smoke tutorial"
read -r -p "Select mode [1]: " MODE
MODE="${MODE:-1}"

echo
echo "Decoder:"
echo "  1) Bayesian latent"
echo "  2) Spectral score"
echo "  3) CCA"
read -r -p "Select decoder [1]: " DECODER_CHOICE
DECODER_CHOICE="${DECODER_CHOICE:-1}"
case "$DECODER_CHOICE" in
  1) DECODER="bayesian_latent" ;;
  2) DECODER="spectral_score" ;;
  3) DECODER="cca" ;;
  *) echo "Invalid decoder choice." >&2; exit 1 ;;
esac

read -r -p "Open realtime GUI? [Y/n]: " GUI_ANSWER
GUI_ANSWER="${GUI_ANSWER:-Y}"
GUI_FLAG="--gui"
if [[ "$GUI_ANSWER" =~ ^[Nn]$ ]]; then
  GUI_FLAG="--no-gui"
fi

echo
echo "Experiment summary"
echo "------------------"
echo "Config:  $CONFIG"
echo "Subject: $SUBJECT"
echo "Mode:    $MODE"
echo "Decoder: $DECODER"
echo "GUI:     $GUI_FLAG"
echo
read -r -p "Start? [Y/n]: " START
START="${START:-Y}"
if [[ "$START" =~ ^[Nn]$ ]]; then
  exit 0
fi

case "$MODE" in
  1)
    python -m bci.cli experiment --config "$CONFIG" --subject "$SUBJECT" --model "$DECODER" "$GUI_FLAG"
    ;;
  2)
    python -m bci.cli evaluate --config "$CONFIG" --subject "$SUBJECT" --model "$DECODER"
    ;;
  3)
    python -m bci.cli bootstrap --config "$CONFIG" --subject "$SUBJECT"
    ;;
  4)
    python -m bci.cli experiment --config "$CONFIG" --subject "$SUBJECT" --model "$DECODER" "$GUI_FLAG" --smoke-mode classifier
    ;;
  5)
    python -m bci.cli experiment --config "$CONFIG" --subject "$SUBJECT" --model "$DECODER" "$GUI_FLAG" --smoke-mode controller
    ;;
  *)
    echo "Invalid mode choice." >&2
    exit 1
    ;;
esac
