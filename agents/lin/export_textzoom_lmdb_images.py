from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

import lmdb
from PIL import Image


DEFAULT_SPLITS = (
    "train1",
    "train2",
    "test/easy",
    "test/medium",
    "test/hard",
)


def clean_filename(text: str, fallback: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    text = text.strip("._")
    return text[:40] or fallback


def read_image(txn: lmdb.Transaction, key: bytes) -> Image.Image:
    raw = txn.get(key)
    if raw is None:
        raise KeyError(key.decode("utf-8", errors="replace"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def save_image(img: Image.Image, path: Path, image_format: str) -> None:
    if image_format == "jpg":
        img.save(path, quality=95)
    else:
        img.save(path)


def export_one_lmdb(lmdb_dir: Path, out_dir: Path, limit: int | None = None, image_format: str = "png") -> int:
    lr_dir = out_dir / "lr"
    hr_dir = out_dir / "hr"
    pair_dir = out_dir / "pairs"
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)
    rows: list[dict[str, str]] = []

    with env.begin(write=False) as txn:
        total_raw = txn.get(b"num-samples")
        if total_raw is None:
            raise RuntimeError(f"No num-samples key found in {lmdb_dir}")
        total = int(total_raw.decode("utf-8"))
        export_total = min(total, limit) if limit is not None else total

        for idx in range(1, export_total + 1):
            key_id = f"{idx:09d}"
            label_raw = txn.get(f"label-{key_id}".encode("ascii"))
            label = label_raw.decode("utf-8", errors="replace") if label_raw else ""
            stem = f"{key_id}_{clean_filename(label, 'no_label')}"

            lr = read_image(txn, f"image_lr-{key_id}".encode("ascii"))
            hr = read_image(txn, f"image_hr-{key_id}".encode("ascii"))

            lr_path = lr_dir / f"{stem}_lr.{image_format}"
            hr_path = hr_dir / f"{stem}_hr.{image_format}"
            pair_path = pair_dir / f"{stem}_pair.{image_format}"
            save_image(lr, lr_path, image_format)
            save_image(hr, hr_path, image_format)

            pair_h = max(lr.height, hr.height)
            gap = 8
            pair = Image.new("RGB", (lr.width + hr.width + gap, pair_h), (245, 245, 245))
            pair.paste(lr, (0, (pair_h - lr.height) // 2))
            pair.paste(hr, (lr.width + gap, (pair_h - hr.height) // 2))
            save_image(pair, pair_path, image_format)

            rows.append(
                {
                    "index": str(idx),
                    "label": label,
                    "lr_path": str(lr_path),
                    "hr_path": str(hr_path),
                    "pair_path": str(pair_path),
                }
            )

    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "lr_path", "hr_path", "pair_path"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def export_default_splits(src_root: Path, out_root: Path, limit: int | None, image_format: str) -> None:
    for split in DEFAULT_SPLITS:
        lmdb_dir = src_root / split
        if not (lmdb_dir / "data.mdb").exists():
            print(f"Skip missing split: {lmdb_dir}")
            continue

        split_out = out_root / split
        count = export_one_lmdb(lmdb_dir, split_out, limit=limit, image_format=image_format)
        print(f"Exported {count} samples: {lmdb_dir} -> {split_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unpack TextZoom LMDB into normal image folders.")
    parser.add_argument("--src-root", type=Path, help="Root containing train1, train2, test/easy, test/medium, test/hard.")
    parser.add_argument("--lmdb", type=Path, help="Path to one LMDB folder, for example workspace/SR/train1.")
    parser.add_argument("--out", required=True, type=Path, help="Output folder.")
    parser.add_argument("--limit", type=int, help="Only export the first N samples from each split.")
    parser.add_argument("--format", choices=("png", "jpg"), default="png", help="Output image format. Default: png.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.src_root:
        export_default_splits(args.src_root, args.out, limit=args.limit, image_format=args.format)
    elif args.lmdb:
        count = export_one_lmdb(args.lmdb, args.out, limit=args.limit, image_format=args.format)
        print(f"Exported {count} samples: {args.lmdb} -> {args.out}")
    else:
        raise SystemExit("Use either --src-root or --lmdb.")


if __name__ == "__main__":
    main()
