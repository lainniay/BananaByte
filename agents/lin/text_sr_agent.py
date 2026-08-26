from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
OUTPUTS = ROOT/"workspace"/"SR"/"AllCharac"/"outputs"

def load_regions(stem):
    path = OUTPUTS / stem / "text_regions" / "union" / "regions.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r",encoding="utf-8") as file:
        data=json.load(file)

    return data["regions"],path

def main():
    stem=input("image stem:").strip()
    regions,path=load_regions(stem)

    print("input:",path)
    print("region count:",len(regions))

if __name__=="__main__":
    main()