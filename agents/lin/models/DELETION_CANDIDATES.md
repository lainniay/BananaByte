# Deletion candidates

No candidate listed here was deleted during the model-folder reorganization. Review the conditions before removing anything.

## Safe generated clutter

These files are generated caches or placeholders and are not source code or experiment evidence.

- `[safe] agents/lin/__pycache__/` — Python bytecode cache, about 0.46 MiB. It includes cache files for scripts that have already moved or been removed.
- `[safe] agents/lin/models/C3-STISR/upstream/model/__pycache__/` — generated Python bytecode, about 0.45 MiB.
- `[safe] agents/lin/models/TPGSR/upstream/model/crnn/__pycache__/` — generated Python bytecode, about 0.03 MiB.
- `[safe] agents/lin/.gitkeep` — the directory is no longer empty, so the placeholder has no remaining purpose. It is tracked by the main repository.

## OmniGen2 deployment-only helpers

These do not participate in normal image generation. They are worth retaining only if the deployment may be rebuilt, moved, or integrity-checked again.

- `[conditional] models/OmniGen2/diagnostics/omnigen2_cuda_smoke.py` — checks CUDA, BF16 support, and a small GPU matrix multiplication.
- `[conditional] models/OmniGen2/diagnostics/omnigen2_snapshot_verify.py` — checks downloaded model shards and hashes without loading them into VRAM.
- `[conditional] models/OmniGen2/maintenance/omnigen2_compare_manifests.py` — compares pre/post-migration snapshot manifests.
- `[conditional] models/OmniGen2/maintenance/relocate_omnigen2_paths.py` — repairs text paths after moving the server-side virtual environment.

The four files above total less than 0.02 MiB. Removing them saves little space; their main cost is maintenance clutter.

## Completed one-off probes

- `[conditional] models/BAGEL/smoke/bagel_smallest.py` — bounded public Hugging Face Space probe. The public Space failed during the earlier test; this file is not used by the current OmniGen2 workflow. Keep it only if that exact probe should be reproducible.
- `[conditional] models/Qwen-Image-Edit/smoke/qwen_smallest.py` — bounded ModelScope API probe that produced the earlier negative Qwen result. It is not used by the current OmniGen2 workflow. Keep it only if the API experiment should be reproducible.

## Large optional repositories

- `[conditional] models/TATT/upstream/` — pristine official TATT checkout, about 30 MiB. No current local wrapper calls it. Delete only if TATT is no longer a planned comparison model.
- `[conditional] models/OmniGen2/upstream/` — pristine official OmniGen2 checkout, about 155 MiB. Current image generation uses the separate server checkout at `/share/linmingheng-local/code/OmniGen2/repo`, so the local checkout is not read at runtime. Keep it for source inspection and version provenance, or delete the entire checkout if the server copy is the accepted source of truth.

## Do not delete as unused

- `models/OmniGen2/inference/`, `scripts/`, and `prompts/` are the active local OmniGen2 workflow.
- `models/C3-STISR/tool/` and `tests/` are active project code and tests.
- `models/C3-STISR/upstream/` contains user modifications and untracked source files; it is not safe to delete as a clean vendor checkout.
- `models/TPGSR/upstream/` is used by the C3-STISR recognizer path. Only its generated `__pycache__/` is a safe cleanup candidate.
