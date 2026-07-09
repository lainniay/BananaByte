from pathlib import Path
from PIL import Image
from agents.lin.metrics import compute_text_errors

from agents.lin.metrics import evaluate, METRIC_DIRECTION
import time

base = Path(__file__).parents[2] / "workspace/SR/AllCharac"
SR_PATH = base / "outputs" /"Canon_002_LR4/baseline/round_3_out.png"  # 任意一张超分输出
HR_PATH = base / "HR" / "Canon_002_LR4.png"               # 同名 HR 真值
# ----------------------------------------------------

sr = Image.open(SR_PATH).convert("RGB")
hr = Image.open(HR_PATH).convert("RGB")
print(f"SR: {SR_PATH.name} {sr.size}   HR: {HR_PATH.name} {hr.size}\n")

# print(">>> 正在评估 (首次运行会下载模型权重, 慢, 耐心等)...\n")
# scores = evaluate(sr, hr, ocr_lang="en")   # 没文字的图可加 run_text=False 跳过 OCR

# print(f"{'指标':<14}{'方向':<6}{'值'}")
# print("-" * 36)
# for name, val in scores.items():
#     arrow = METRIC_DIRECTION.get(name, "")
#     shown = f"{val:.4f}" if isinstance(val, float) else str(val)
#     print(f"{name:<14}{arrow:<6}{shown}")

te = compute_text_errors(sr, hr, lang="ch")   # 跟你 evaluate 用的语言一致
print("HR 文字:", repr(te["ref_text"]))
print("SR 文字:", repr(te["hyp_text"]))
print("CER:", te["cer"], " WER:", te["wer"])


