#!/usr/bin/env bash
set -euo pipefail

CALL_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -ne 1 ]]; then
  echo "Uso: ./launcher_synthetic.sh <config-json|nombre-config>" >&2
  echo "Ejemplos:" >&2
  echo "  ./launcher_synthetic.sh laucher_synthetics" >&2
  echo "  ./launcher_synthetic.sh ../../configs/laucher_synthetics.json" >&2
  exit 1
fi

CONFIG_INPUT="$1"

if [[ "$CONFIG_INPUT" == */* || "$CONFIG_INPUT" == *.json ]]; then
  if [[ "$CONFIG_INPUT" = /* ]]; then
    CONFIG_PATH="$CONFIG_INPUT"
  elif [[ -f "${CALL_DIR}/${CONFIG_INPUT}" ]]; then
    CONFIG_PATH="${CALL_DIR}/${CONFIG_INPUT}"
  else
    CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_INPUT}"
  fi
else
  CONFIG_PATH="${PROJECT_DIR}/configs/${CONFIG_INPUT}.json"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "No existe el archivo de configuración: $CONFIG_PATH" >&2
  echo "PWD original: $CALL_DIR" >&2
  echo "SCRIPT_DIR: $SCRIPT_DIR" >&2
  exit 1
fi

pixi run python launcher_synthetic.py --config "$CONFIG_PATH"
