#!/usr/bin/env bash
set -e
GPUS="${GPUS:-4,5,6,7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RM=${STAP_ROOT}/rewardmodel
ENV_PY=${PYTHON_BIN:-${CONDA_BASE:-$HOME/miniconda3}/envs/stap/bin/python}
LOG=${RM}/logs; mkdir -p "${LOG}"

export CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${STAP_ROOT}/models/clip-vit-large-patch14}"
export HPS_MODEL_DIR="${HPS_MODEL_DIR:-${STAP_ROOT}/models/HPSv2}"

CLIP_GPU="${CLIP_GPU:-6}";  CLIP_PORT="${CLIP_SERVER_PORT:-8030}"
HPS_GPU="${HPS_GPU:-7}";    HPS_PORT="${HPS_SERVER_PORT:-8020}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,::1,${no_proxy:-}"

_health () { curl -s --noproxy '*' "localhost:$1/health" >/dev/null 2>&1; }

if ! "${ENV_PY}" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "[reward-serve] installing fastapi + uvicorn into the STAP env ..."
    "${ENV_PY}" -m pip install -q fastapi uvicorn
fi

[ -d "${CLIP_MODEL_PATH}" ] || { echo "ERROR: CLIP_MODEL_PATH not a dir: ${CLIP_MODEL_PATH}"; exit 1; }

HPS_OK=1
if [ ! -f "${HPS_MODEL_DIR}/HPS_v2.1_compressed.pt" ]; then
    HPS_OK=0
    echo "WARN: ${HPS_MODEL_DIR}/HPS_v2.1_compressed.pt not found -> SKIPPING HPS server (r_HPS=0)."
    echo "      To enable r_HPS, copy the weights into place, e.g.:"
    echo "          mkdir -p ${STAP_ROOT}/models && \\"
    echo "          cp -r \$HPSV2_SOURCE_DIR ${STAP_ROOT}/models/"
    echo "      then re-run this script."
fi

if _health "${CLIP_PORT}"; then
    echo "[CLIP] already running on :${CLIP_PORT}"
else
    echo "[CLIP] starting on GPU ${CLIP_GPU}, port ${CLIP_PORT} (model: ${CLIP_MODEL_PATH}) ..."
    CUDA_VISIBLE_DEVICES="${CLIP_GPU}" CLIP_SERVER_PORT="${CLIP_PORT}" \
        nohup "${ENV_PY}" "${RM}/clip_server.py" > "${LOG}/clip_${CLIP_PORT}.log" 2>&1 &
fi

if [ "${HPS_OK}" = 1 ]; then
    if _health "${HPS_PORT}"; then
        echo "[HPS] already running on :${HPS_PORT}"
    else
        echo "[HPS] starting on GPU ${HPS_GPU}, port ${HPS_PORT} ..."
        CUDA_VISIBLE_DEVICES="${HPS_GPU}" HPS_SERVER_PORT="${HPS_PORT}" \
            nohup "${ENV_PY}" "${RM}/hps_server.py" > "${LOG}/hps_${HPS_PORT}.log" 2>&1 &
    fi
fi

echo "[reward-serve] waiting for health (model load ~30-60s) ..."
for i in $(seq 1 90); do
    clip_ready=0; hps_ready=0
    _health "${CLIP_PORT}" && clip_ready=1
    if [ "${HPS_OK}" = 1 ]; then _health "${HPS_PORT}" && hps_ready=1; else hps_ready=1; fi
    if [ "${clip_ready}" = 1 ] && [ "${hps_ready}" = 1 ]; then
        echo "[reward-serve] READY:"
        echo "    export CLIP_SERVER_URL=localhost:${CLIP_PORT}"
        if [ "${HPS_OK}" = 1 ]; then
            echo "    export HPS_SERVER_URL=localhost:${HPS_PORT}"
        else
            echo "    (HPS skipped - r_HPS will be 0 until the weight is downloaded)"
        fi
        echo "(CLIP_SERVER_URL / HPS_SERVER_URL are already defaults in run_STAP_grpo.sh)"
        exit 0
    fi
    sleep 2
done
echo "[reward-serve] TIMEOUT waiting for health. Check logs:"
echo "    tail -f ${LOG}/clip_${CLIP_PORT}.log ${LOG}/hps_${HPS_PORT}.log"
exit 1

