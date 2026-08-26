"""Run C3-STISR on one text crop.

This is a standalone experiment script for Lin's SR workflow. It does not touch
the shared state machine. The first goal is to prove that the downloaded
C3-STISR checkpoint can restore one LR text crop end to end.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


@contextlib.contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old_cwd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-crop C3-STISR inference.")
    parser.add_argument("--input", required=True, type=Path, help="Input LR text crop image.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="C3-STISR .pth checkpoint.")
    parser.add_argument(
        "--recognizer-checkpoint",
        type=Path,
        default=None,
        help="CRNN checkpoint. Defaults to recognizer_best_0.pth beside --checkpoint.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output SR image path.")
    parser.add_argument(
        "--c3-root",
        type=Path,
        default=Path("agents/lin/models/C3-STISR/upstream"),
        help="Path to the C3-STISR repository root.",
    )
    parser.add_argument(
        "--tpgsr-root",
        type=Path,
        default=Path("agents/lin/models/TPGSR/upstream"),
        help="TPGSR repository root containing the CRNN definition used by C3-STISR.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("none", "rec", "rec-ling"),
        default="rec-ling",
        help="C3 clue path. rec-ling reproduces the released full inference path.",
    )
    parser.add_argument("--input-width", type=int, default=64, help="Model input crop width.")
    parser.add_argument("--input-height", type=int, default=16, help="Model input crop height.")
    parser.add_argument("--output-width", type=int, default=128, help="Model output width.")
    parser.add_argument("--output-height", type=int, default=32, help="Model output height.")
    parser.add_argument(
        "--save-mode",
        choices=("tanh", "clamp"),
        default="clamp",
        help="Convert model output to image. clamp matches C3's [0,1] training targets; tanh is a legacy comparison.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--verbose-import",
        action="store_true",
        help="Show noisy debug prints emitted by the released C3 code during import.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def resolve_from_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def load_c3_module(c3_root: Path, verbose_import: bool):
    """Import C3-STISR from its repo root.

    The released code reads model/charset.txt through a relative path during
    import, so we temporarily chdir into the C3 repo.
    """
    c3_root = c3_root.resolve()
    sys.path.insert(0, str(c3_root))
    with pushd(c3_root):
        if verbose_import:
            from model import c3  # type: ignore
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                from model import c3  # type: ignore
    return c3


def build_model(c3_module):
    return c3_module.TSRN_TL(
        scale_factor=2,
        width=128,
        height=32,
        STN=True,
        srb_nums=5,
        mask=True,
        hidden_units=32,
        text_emb=37,
        out_text_channels=32,
        triple_clues=True,
    )


def load_recognizer(tpgsr_root: Path, checkpoint_path: Path, device):
    import torch

    crnn_path = tpgsr_root.resolve() / "model" / "crnn" / "crnn.py"
    if not crnn_path.is_file():
        raise FileNotFoundError(f"TPGSR CRNN definition not found: {crnn_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CRNN checkpoint not found: {checkpoint_path}")

    spec = importlib.util.spec_from_file_location("c3stisr_tpgsr_crnn", crnn_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CRNN module from {crnn_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    recognizer = module.CRNN(32, 1, 37, 256).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    recognizer.load_state_dict(strip_module_prefix(state_dict), strict=True)
    recognizer.eval()
    return recognizer


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def crnn_input_from_c3_tensor(lr_tensor):
    import torch.nn.functional as functional

    rgb = functional.interpolate(lr_tensor[:, :3], (32, 100), mode="bicubic", align_corners=False)
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def infer_with_priors(model, lr_tensor, prior_mode: str, recognizer=None):
    import torch

    if prior_mode == "none":
        return model(lr_tensor), {}
    if recognizer is None:
        raise ValueError(f"prior_mode={prior_mode!r} requires a recognizer")

    rec_logits = recognizer(crnn_input_from_c3_tensor(lr_tensor))
    rec_probs_tbc = torch.softmax(rec_logits, dim=-1)
    rec_clue = rec_probs_tbc.permute(1, 2, 0).unsqueeze(2)
    clues = {"rec": rec_probs_tbc.permute(1, 0, 2)}

    if prior_mode == "rec":
        return model(lr_tensor, rec_clue), clues
    if prior_mode != "rec-ling":
        raise ValueError(f"Unsupported prior mode: {prior_mode}")

    lm_input = rec_clue.squeeze(2).transpose(1, 2).detach()
    lengths = torch.full(
        (lm_input.size(0),),
        lm_input.size(1),
        dtype=torch.long,
        device=lr_tensor.device,
    )
    ling_probs_btc = torch.softmax(model.lm(lm_input, lengths)["logits"], dim=-1)
    ling_clue = ling_probs_btc.unsqueeze(2).transpose(1, 3)
    clues["ling"] = ling_probs_btc
    return model(lr_tensor, rec_clue, ling_clue, None), clues


def decode_ctc_probabilities(probabilities: np.ndarray) -> list[str]:
    alphabet = "-0123456789abcdefghijklmnopqrstuvwxyz"
    decoded: list[str] = []
    for sequence in probabilities.argmax(axis=-1):
        previous = -1
        characters: list[str] = []
        for index in sequence.tolist():
            if index != 0 and index != previous:
                characters.append(alphabet[index])
            previous = index
        decoded.append("".join(characters))
    return decoded


def clue_debug(clues: dict[str, object]) -> dict[str, object]:
    debug: dict[str, object] = {}
    for name, tensor in clues.items():
        probabilities = tensor.detach().cpu().numpy()
        debug[name] = {
            "shape": list(probabilities.shape),
            "decoded": decode_ctc_probabilities(probabilities),
            "mean_max_probability": probabilities.max(axis=-1).mean(axis=-1).tolist(),
        }
    return debug


def image_to_tensor(image: Image.Image, width: int, height: int):
    import torch

    image = image.convert("RGB").resize((width, height), Image.BICUBIC)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    threshold = float(gray.mean())
    mask = (gray <= threshold).astype(np.float32)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)

    return torch.cat((rgb_tensor, mask_tensor), dim=0).unsqueeze(0)


def tensor_to_image(tensor, save_mode: str) -> Image.Image:
    if save_mode == "tanh":
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    args = parse_args()

    input_path = resolve_from_repo(args.input)
    checkpoint_path = resolve_from_repo(args.checkpoint)
    output_path = resolve_from_repo(args.output)
    c3_root = resolve_from_repo(args.c3_root)
    tpgsr_root = resolve_from_repo(args.tpgsr_root)
    recognizer_checkpoint = (
        resolve_from_repo(args.recognizer_checkpoint)
        if args.recognizer_checkpoint
        else checkpoint_path.with_name("recognizer_best_0.pth")
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not c3_root.is_dir():
        raise FileNotFoundError(f"C3-STISR repo not found: {c3_root}")

    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    c3_module = load_c3_module(c3_root, args.verbose_import)
    with pushd(c3_root):
        model = build_model(c3_module).to(device)

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        state_dict = checkpoint["state_dict_G"] if isinstance(checkpoint, dict) and "state_dict_G" in checkpoint else checkpoint
        model.load_state_dict(strip_module_prefix(state_dict), strict=True)
        model.eval()

    recognizer = None
    if args.prior_mode != "none":
        recognizer = load_recognizer(tpgsr_root, recognizer_checkpoint, device)

    image = Image.open(input_path)
    lr_tensor = image_to_tensor(image, args.input_width, args.input_height).to(device)

    with torch.no_grad():
        sr_tensor, clues = infer_with_priors(model, lr_tensor, args.prior_mode, recognizer)

    sr_rgb = sr_tensor[0, :3, :, :]
    sr_image = tensor_to_image(sr_rgb, args.save_mode)
    if sr_image.size != (args.output_width, args.output_height):
        sr_image = sr_image.resize((args.output_width, args.output_height), Image.BICUBIC)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr_image.save(output_path)
    debug_path = output_path.with_name(output_path.stem + "_clues.json")
    debug_path.write_text(
        json.dumps(
            {
                "prior_mode": args.prior_mode,
                "recognizer_checkpoint": str(recognizer_checkpoint) if recognizer else None,
                "clues": clue_debug(clues),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"device: {device}")
    print(f"input: {input_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"prior mode: {args.prior_mode}")
    if recognizer is not None:
        print(f"recognizer: {recognizer_checkpoint}")
    print(f"output: {output_path}")
    print(f"clues: {debug_path}")
    print(f"raw output tensor: {tuple(sr_tensor.shape)}")
    print(f"raw output min/max: {float(sr_tensor.min()):.6f} / {float(sr_tensor.max()):.6f}")


if __name__ == "__main__":
    main()
