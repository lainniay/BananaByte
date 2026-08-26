"""Build LR / VLM output / HR triptychs for selected_LR."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from agents.lin.selected_lr_vlm_batch import find_hr


PANEL = 500
HEADER = 54
GAP = 12


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()


def panel(image: Image.Image, label: str, nearest: bool = False) -> Image.Image:
    canvas = Image.new("RGB", (PANEL, PANEL + HEADER), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"{label}  {image.width}x{image.height}", fill="black", font=font(26))
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (PANEL, PANEL),
        Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS,
    )
    x = (PANEL - fitted.width) // 2
    y = HEADER + (PANEL - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def triptych(lr: Image.Image, generated: Image.Image, hr: Image.Image) -> Image.Image:
    parts = [panel(lr, "LR", True), panel(generated, "VLM SR"), panel(hr, "HR")]
    result = Image.new("RGB", (PANEL * 3 + GAP * 2, PANEL + HEADER), "#dddddd")
    for index, part in enumerate(parts):
        result.paste(part, (index * (PANEL + GAP), 0))
    return result


def contact_sheet(items: list[tuple[str, Image.Image]], path: Path) -> None:
    thumb_width = 900
    thumbs = []
    for name, image in items:
        resized = ImageOps.contain(image, (thumb_width, 380), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_width, resized.height + 40), "white")
        ImageDraw.Draw(tile).text((8, 6), name, fill="black", font=font(24))
        tile.paste(resized, ((thumb_width - resized.width) // 2, 40))
        thumbs.append(tile)

    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    row_height = max(tile.height for tile in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * row_height), "#cccccc")
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % columns) * thumb_width, (index // columns) * row_height))
    sheet.save(path)


def main() -> None:
    root = Path(__file__).parents[2] / "workspace/SR/COLLECT"
    input_dir = root / "selected_LR"
    output_root = input_dir / "vlm_outputs"
    comparison_dir = output_root / "comparisons"
    individual_dir = comparison_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[int, list[tuple[str, Image.Image]]] = {128: [], 256: [], 512: []}
    manifest = []
    for lr_path in sorted(input_dir.glob("*.png")):
        item_dir = output_root / lr_path.stem
        metadata_path = item_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        generated_path = Path(metadata["output"])
        hr_path = find_hr(root / "HR", lr_path)
        with Image.open(lr_path) as lr, Image.open(generated_path) as generated, Image.open(hr_path) as hr:
            comparison = triptych(lr, generated, hr)
            individual_path = individual_dir / f"{lr_path.stem}_triptych.png"
            comparison.save(individual_path)
            size_group = max(lr.size)
            groups.setdefault(size_group, []).append((lr_path.stem, comparison.copy()))
            manifest.append({"lr": str(lr_path), "generated": str(generated_path), "hr": str(hr_path), "comparison": str(individual_path)})

    all_items = []
    for size_group, items in sorted(groups.items()):
        if items:
            contact_sheet(items, comparison_dir / f"comparison_LR{size_group}.png")
            all_items.extend(items)
    if all_items:
        contact_sheet(all_items, comparison_dir / "comparison_all.png")
    (comparison_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
