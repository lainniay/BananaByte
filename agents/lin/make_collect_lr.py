import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps


LEVELS = {
    "LR_128": 128,
    "LR_256": 256,
    "LR_512": 512,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main():
    default_collect = Path(__file__).resolve().parents[2] / "workspace" / "SR" / "COLLECT"
    parser = argparse.ArgumentParser(description="Create graded LR images from COLLECT/HR.")
    parser.add_argument("--collect-dir", type=Path, default=default_collect)
    args = parser.parse_args()

    hr_dir = args.collect_dir / "HR"
    sources = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not sources:
        raise SystemExit(f"No images found in {hr_dir}")

    rows = []
    for source in sources:
        with Image.open(source) as opened:
            hr = ImageOps.exif_transpose(opened).convert("RGB")

        hr_width, hr_height = hr.size
        for level, long_side in LEVELS.items():
            lr_dir = args.collect_dir / level
            lr_dir.mkdir(parents=True, exist_ok=True)

            scale = long_side / max(hr_width, hr_height)
            lr_size = (
                max(1, round(hr_width * scale)),
                max(1, round(hr_height * scale)),
            )
            output = lr_dir / f"{source.stem}.png"
            if output.exists():
                try:
                    with Image.open(output) as existing:
                        if existing.size == lr_size:
                            rows.append(
                                [source.name, level, long_side, hr_width, hr_height, *lr_size, output.name]
                            )
                            continue
                except OSError:
                    pass

            hr.resize(lr_size, Image.Resampling.BICUBIC).save(output, compress_level=1)
            rows.append(
                [source.name, level, long_side, hr_width, hr_height, *lr_size, output.name]
            )

    manifest = args.collect_dir / "lr_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["source", "level", "long_side", "hr_width", "hr_height", "lr_width", "lr_height", "output"]
        )
        writer.writerows(rows)

    print(f"HR images: {len(sources)}")
    print(f"LR images: {len(rows)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
