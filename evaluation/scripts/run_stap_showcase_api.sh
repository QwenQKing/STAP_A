#!/usr/bin/env bash
set -euo pipefail

if [ "$(basename "${CONDA_PREFIX:-}")" != "stap" ]; then
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-stap}"
fi

export PYTHONUNBUFFERED=1
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAN22_BASE_URLS="${WAN22_BASE_URLS:-}"

ARGS=("$@")
has_wan_urls=0
for arg in "${ARGS[@]}"; do
  if [ "$arg" = "--wan22-base-urls" ]; then
    has_wan_urls=1
    break
  fi
done

if [ "$has_wan_urls" -eq 0 ]; then
  if [ -z "$WAN22_BASE_URLS" ]; then
    echo "Set WAN22_BASE_URLS or pass --wan22-base-urls." >&2
    exit 1
  fi
  read -r -a DEFAULT_WAN22_ENDPOINTS <<< "$WAN22_BASE_URLS"
  ARGS=(--wan22-base-urls "${DEFAULT_WAN22_ENDPOINTS[@]}" "${ARGS[@]}")
fi

python3 "${SCRIPT_DIR}/run_stap_showcase_api.py" "${ARGS[@]}"

