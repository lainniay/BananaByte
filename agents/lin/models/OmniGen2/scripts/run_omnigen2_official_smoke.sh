#!/usr/bin/env bash
set -euo pipefail

deploy_root=/share/linmingheng-local/code/OmniGen2
repo_dir=${deploy_root}/repo
venv_dir=${deploy_root}/.venv
model_dir=${deploy_root}/models/OmniGen2
run_dir=${deploy_root}/runs/official_edit_seed0
input_path=${repo_dir}/example_images/ComfyUI_temp_mllvz_00071_.png
output_path=${run_dir}/output.png

if [[ -e "${run_dir}" ]]; then
  echo "Run directory already exists: ${run_dir}" >&2
  exit 2
fi

mkdir -p "${run_dir}"
date --iso-8601=seconds > "${run_dir}/started_at.txt"
sha256sum "${input_path}" > "${run_dir}/input.sha256"
git -C "${repo_dir}" rev-parse HEAD > "${run_dir}/code_commit.txt"
"${venv_dir}/bin/pip" freeze --all > "${run_dir}/requirements.freeze.txt"
nvidia-smi -i 1 -q > "${run_dir}/gpu_before.txt"

nvidia-smi \
  --query-gpu=memory.used \
  --format=csv,noheader,nounits \
  -i 1 \
  -lms 500 > "${run_dir}/vram_mib.log" &
monitor_pid=$!

cleanup_monitor() {
  if kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}
trap cleanup_monitor EXIT

set +e
/usr/bin/time -v -o "${run_dir}/time.txt" \
  env \
  CPATH="${deploy_root}/sysroot/usr/include:${deploy_root}/sysroot/usr/include/python3.10" \
  PYTHONPATH="${repo_dir}" \
  CUDA_VISIBLE_DEVICES=1 \
  "${venv_dir}/bin/python" "${repo_dir}/inference.py" \
    --model_path "${model_dir}" \
    --dtype bf16 \
    --scheduler euler \
    --num_inference_step 50 \
    --seed 0 \
    --height 1024 \
    --width 1024 \
    --text_guidance_scale 5.0 \
    --image_guidance_scale 2.0 \
    --cfg_range_start 0.0 \
    --cfg_range_end 1.0 \
    --instruction "Change the background to classroom." \
    --input_image_path "${input_path}" \
    --output_image_path "${output_path}" \
    --num_images_per_prompt 1 \
    > "${run_dir}/stdout.log" \
    2> "${run_dir}/stderr.log"
run_status=$?
set -e

cleanup_monitor
trap - EXIT

date --iso-8601=seconds > "${run_dir}/finished_at.txt"
printf '%s\n' "${run_status}" > "${run_dir}/exit_code.txt"
awk 'BEGIN { max = 0 } { if ($1 + 0 > max) max = $1 + 0 } END { print max }' \
  "${run_dir}/vram_mib.log" > "${run_dir}/peak_vram_mib.txt"
nvidia-smi -i 1 -q > "${run_dir}/gpu_after.txt"

exit "${run_status}"
