# OmniGen2 local integration

## Active workflow

- `inference/omnigen2_scaled_inference.py`: server-side thin wrapper around the official pipeline. Supports direct 4x generation and bicubic pre-upsample plus aligned generation.
- `scripts/run_omnigen2.ps1`: single-image local-to-server runner. Uploads an input and prompt, runs inference, downloads the complete remote run, and optionally opens the output.
- `scripts/run_omnigen2_ablation.ps1`: fixed four-configuration experiment runner for direct/pre-upsample and semantic/conservative prompts.
- `scripts/make_sr_triptych.ps1`: exact Input-Output-HR comparison composer; it does not use generative editing.
- `scripts/run_omnigen2_official_smoke.sh`: official-example deployment smoke test with timing and VRAM logging.
- `prompts/`: OmniGen2-specific restoration prompts.
- `upstream/`: official OmniGen2 Git checkout. Do not edit it for project-specific behavior.

## Retained diagnostics and maintenance

- `diagnostics/omnigen2_cuda_smoke.py`: checks CUDA visibility, BF16 support, and a small BF16 matrix multiplication. Useful after rebuilding or moving the environment.
- `diagnostics/omnigen2_snapshot_verify.py`: verifies model snapshot shards and hashes without loading weights into VRAM. Useful after model download or migration.
- `maintenance/omnigen2_compare_manifests.py`: compares snapshot manifests produced before and after a migration.
- `maintenance/relocate_omnigen2_paths.py`: repairs text paths inside a moved virtual environment and bootstrap directory.

The diagnostic and maintenance files are not part of normal image generation. If the current deployment will no longer be moved, rebuilt, or integrity-checked, they are deletion candidates; delete them only with user approval.
