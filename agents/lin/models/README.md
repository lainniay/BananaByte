# Model directories

Model-specific code belongs under one self-contained directory per model.

- `OmniGen2/`: local inference wrappers, prompts, run scripts, diagnostics, maintenance tools, and the official repository.
- `BAGEL/`: BAGEL-specific probes.
- `Qwen-Image-Edit/`: Qwen-Image-Edit-specific probes.
- `C3-STISR/`: C3-STISR tools, tests, and official repository.
- `TPGSR/`: official TPGSR repository used by C3-STISR.
- `TATT/`: official TATT repository.

Generic dataset preparation, OCR, evaluation, and Agent orchestration remain in `agents/lin/`.
