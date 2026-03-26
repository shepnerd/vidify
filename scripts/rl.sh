#!/usr/bin/env bash
set -euo pipefail

# Configure these via environment variables or edit defaults below.
CHARGED_GROUP_DEFAULT="${RL_CHARGED_GROUP:-my_gpu_group}"
PRIVATE_MACHINE_DEFAULT="${RL_PRIVATE_MACHINE:-group}"

# Shared filesystem mount (adjust for your cluster)
MOUNT_ARGS=()
if [[ -n "${RL_MOUNT:-}" ]]; then
  MOUNT_ARGS=("--mount=${RL_MOUNT}")
fi

RDMA_ARGS=(
  "--custom-resources" "rdma/mlnx_shared=8"
  "--custom-resources" "mellanox.com/mlnx_rdma=1"
)

usage() {
  cat <<'EOF'
Usage:
  rl [-gpu N] [-cpu N] [-mem MB] [-image IMG] [-d] [--] [CMD...]

Defaults:
  If gpu > 0:
    cpu = gpu * 4
    mem = gpu * 16384 (MB, 16GB per GPU)
  If gpu = 0 (CPU-only):
    cpu = 4
    mem = 16384 (MB)

Behavior:
  - gpu<=8: -P 1, --gpu=gpu
  - gpu>8 : must be multiple of 8, auto -P=gpu/8 and --gpu=8 per replica
  - If no CMD: start detached worker with "sleep inf" (recommended; SSH in from UI)
  - -d not allowed with multi-node (-P>1)

Examples:
  rl -gpu 0
  rl -gpu 0 -- bash -c 'python -V'
  rl -gpu 8
EOF
}

GPU="0"
CPU=""
MEM_MB=""
IMAGE=""
DETACH="false"
CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -gpu) GPU="${2:-}"; shift 2;;
    -cpu) CPU="${2:-}"; shift 2;;
    -mem) MEM_MB="${2:-}"; shift 2;;
    -image|--image) IMAGE="${2:-}"; shift 2;;
    -d|--detach) DETACH="true"; shift;;
    -h|--help) usage; exit 0;;
    --) shift; CMD=("$@"); break;;
    *) CMD+=("$1"); shift;;
  esac
done

[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "ERROR: -gpu must be >=0 integer" >&2; exit 1; }

# defaults (gpu-aware)
if [[ -z "${CPU}" ]]; then
  if (( GPU == 0 )); then CPU=4; else CPU=$(( GPU * 4 )); fi
fi
if [[ -z "${MEM_MB}" ]]; then
  if (( GPU == 0 )); then MEM_MB=16384; else MEM_MB=$(( GPU * 16384 )); fi
fi

# split replicas
PER_NODE=8
NODES=1
GPU_PER_NODE="${GPU}"
if (( GPU > PER_NODE )); then
  (( GPU % PER_NODE == 0 )) || { echo "ERROR: -gpu > 8 must be multiple of 8 (got ${GPU})" >&2; exit 1; }
  NODES=$(( GPU / PER_NODE ))
  GPU_PER_NODE=$PER_NODE
fi

# default command: detached + sleep
if [[ ${#CMD[@]} -eq 0 ]]; then
  CMD=(bash -c "sleep inf")
  DETACH="true"
fi

if [[ "${DETACH}" == "true" ]] && (( NODES > 1 )); then
  echo "ERROR: multi-node (-P>1) does not support -d/--detach" >&2
  exit 1
fi

DETACH_ARGS=()
[[ "${DETACH}" == "true" ]] && DETACH_ARGS=(-d)

IMAGE_ARGS=()
[[ -n "${IMAGE}" ]] && IMAGE_ARGS=(--image "${IMAGE}")

# GPU-only flags
GPU_FLAGS=()
EXTRA_ARGS=()
if (( GPU > 0 )); then
  GPU_FLAGS+=(--gpu="${GPU_PER_NODE}" --private-machine="${PRIVATE_MACHINE_DEFAULT}")
  if (( NODES > 1 )); then
    EXTRA_ARGS+=("${RDMA_ARGS[@]}")
  fi
else
  GPU_FLAGS+=(--gpu=0)
fi

exec rlaunch \
  "${GPU_FLAGS[@]}" \
  --memory="${MEM_MB}" \
  --cpu="${CPU}" \
  --charged-group="${CHARGED_GROUP_DEFAULT}" \
  -P "${NODES}" \
  --enable-sshd=true \
  "${IMAGE_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  -e DISTRIBUTED_JOB=true \
  "${DETACH_ARGS[@]}" \
  -- "${CMD[@]}"