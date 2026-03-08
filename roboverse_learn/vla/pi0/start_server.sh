#!/bin/bash
# Start π₀ policy server for RoboVerse evaluation
# Usage: bash roboverse_learn/vla/pi0/start_server.sh [config_name]
#
# Pre-requisite: Install openpi first:
#   cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync
#
# Download weights (pi0_libero for zero-shot LIBERO eval):
#   .venv311/bin/gsutil -m cp -r \
#     gs://openpi-assets/checkpoints/pi0_libero \
#     third_party/openpi/checkpoints/pi0_libero

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROBOVERSE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENPI_ROOT="${ROBOVERSE_ROOT}/third_party/openpi"

config_name="${1:-pi0_roboverse_lora}"
ckpt_dir="${CKPT_DIR:-${OPENPI_ROOT}/checkpoints/pi0_libero}"

if [ ! -d "${OPENPI_ROOT}" ]; then
  echo "❌ openpi not found at ${OPENPI_ROOT}"
  echo "   Run: cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync"
  exit 1
fi

if [ ! -d "${ckpt_dir}" ]; then
  echo "❌ Checkpoint not found: ${ckpt_dir}"
  echo "   Download weights:"
  echo "   .venv311/bin/gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base third_party/openpi/checkpoints/pi0_base"
  exit 1
fi

# Register roboverse_policy if not already done
POLICY_DEST="${OPENPI_ROOT}/src/openpi/policies/roboverse_policy.py"
if [ ! -f "${POLICY_DEST}" ]; then
  echo "📋 Registering roboverse_policy.py..."
  cp "${ROBOVERSE_ROOT}/roboverse_learn/vla/pi0/roboverse_policy.py" "${POLICY_DEST}"
  echo "✅ roboverse_policy.py registered"
fi

echo "🚀 Starting π₀ server..."
echo "   Config: ${config_name}"
echo "   Checkpoint: ${ckpt_dir}"
echo "   (Connect eval at localhost:8000)"

cd "${OPENPI_ROOT}"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="${config_name}" \
  --policy.dir="${ckpt_dir}"
