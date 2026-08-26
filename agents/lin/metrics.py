import numpy as np
from PIL import Image
from agents.lin.fidelity import image_to_array, low_pass
import time

# ============================================================
# 指标方向速查表: ↑=越大越好, ↓=越小越好
# ============================================================
METRIC_DIRECTION = {
      "psnr": "↑",          # 保真度
      "ssim": "↑",          # 保真度
      "lpips": "↓",         # 感知 (越小越像)
      "dists": "↓",         # 感知 (越小越像)
      "maniqa": "↑",        # 感知 (无参考画质)
      "musiq": "↑",         # 感知 (无参考画质)
      "cer": "↓",           # 文字: 字符错误率
      "wer": "↓",           # 文字: 词错误率
      "text_halluc": "↓",   # 语义: 文字幻觉标志 (0/1)
  }

#懒加载缓存（避免重复创建模型）
_DEVICE=None
_metric_cache:dict={} #pyiqa指标对象缓存
_ocr_cache:dict={} #PaddleOCR 缓存

def _device() -> str:
    """返回可用设备, 有 GPU 用 cuda 否则 cpu."""
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def _pil_to_tensor(img:Image.Image):
    """PTL image->pyiqa 需要的张量(1,3,H,W),值域[0,1]."""
    import torch
    arr=np.array(img.convert("RGB"),dtype=np.float32)/255.0
    t=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return t.to(_device())

def _get_metric(name:str,**kwargs):
    """创建/复用一个 pyiqa指标对象."""
    key=name+str(sorted(kwargs.items()))
    if key not in _metric_cache:
        import pyiqa
        _metric_cache[key]=pyiqa.create_metric(name,**kwargs).to(_device())
    return _metric_cache[key]

def _fr_score(name:str,sr:Image.Image,hr:Image.Image,**kwargs)->float:
    """全参考指标,需要HR,尺寸必须一致."""
    metric=_get_metric(name,**kwargs)
    return float(metric(_pil_to_tensor(sr),_pil_to_tensor(hr)).item())\

def _nr_score(name:str,sr:Image.Image,**kwargs)->float:
    """无参考指标."""
    metric=_get_metric(name,**kwargs)
    return float(metric(_pil_to_tensor(sr)).item())

# ============================================================
# 一、Fidelity 保真度 (需要 HR)
# ============================================================

def compute_psnr(sr: Image.Image, hr: Image.Image) -> float:
    """PSNR (峰值信噪比, Y 通道)  方向: ↑ 越大越好.

    怎么算: 在 YCbCr 的 Y(亮度)通道上, 对 SR 与 HR 求均方误差(MSE),
    再换算成分贝 PSNR = 10*log10(255^2 / MSE)。
    原理: 纯像素级误差。误差越小 PSNR 越高。只看亮度通道是超分论文惯例
    (人眼对亮度比对颜色敏感)。值越高=越逐像素接近真值, 但不代表好看.
    """
    return _fr_score("psnr", sr, hr, test_y_channel=True, color_space="ycbcr")


def compute_ssim(sr: Image.Image, hr: Image.Image) -> float:
    """SSIM (结构相似性, Y 通道)  方向: ↑ 越大越好, 范围 [-1, 1].

    怎么算: 在局部窗口里比较两图的亮度均值、对比度(方差)、结构(协方差),
    综合成一个相似度分数, 全图取平均。
    原理: 比 PSNR 更贴近人眼——人看的是"结构"而非单个像素差。
    1=结构完全一致。
    """
    return _fr_score("ssim", sr, hr, test_y_channel=True)


# ============================================================
# 二、Perception 感知质量
# ============================================================
def compute_lpips(sr: Image.Image, hr: Image.Image) -> float:
    """LPIPS (学习感知图像块相似度)  方向: ↓ 越小越好.

    怎么算: 把 SR 和 HR 都喂进一个预训练 CNN(AlexNet/VGG), 比较深层特征的距离。
    原理: 用神经网络特征代替像素来度量"感知差异", 和人类判断高度吻合。
    0=感知上完全一样。是感知质量的主流指标。
    """
    return _fr_score("lpips", sr, hr)


def compute_dists(sr: Image.Image, hr: Image.Image) -> float:
    """DISTS (深度图像结构与纹理相似度)  方向: ↓ 越小越好.

    怎么算: 同样基于深度特征, 但同时建模"结构"和"纹理"两部分相似度。
    原理: 对纹理类内容(草地、毛发)比 LPIPS 更宽容、更稳。0=最相似。
    """
    return _fr_score("dists", sr, hr)


def compute_maniqa(sr: Image.Image) -> float:
    """MANIQA (无参考画质评分)  方向: ↑ 越大越好.

    怎么算: 一个在人工打分数据上训练的 Transformer 模型, 直接给 SR 打质量分。
    原理: 无参考(不需要 HR), 模拟人类对单张图的画质打分。部署时可用。
    """
    return _nr_score("maniqa", sr)


def compute_musiq(sr: Image.Image) -> float:
    """MUSIQ (多尺度无参考画质评分)  方向: ↑ 越大越好.

    怎么算: 多尺度 Transformer 直接对单张图打分, 兼顾全局和局部质量。
    原理: 同为无参考画质指标, 视角和 MANIQA 互补, 常一起报告。
    """
    return _nr_score("musiq", sr)


# ============================================================
# 三、Semantic 语义一致性
# ============================================================

def _ocr_text(img: Image.Image, lang: str = "en") -> str:
    """用 PaddleOCR 提取图中文字, 拼成一个字符串。失败返回空串.

    兼容 PaddleOCR 3.x (新 API) 与 2.x (旧 API)。lang='en' 英文, 'ch' 中文。
    """
    if lang not in _ocr_cache:
        from paddleocr import PaddleOCR
        # 3.x: use_textline_orientation 取代旧的 use_angle_cls; 不再有 show_log
        t0 = time.time()

        _ocr_cache[lang] = PaddleOCR(lang=lang, use_textline_orientation=True,enable_mkldnn=False,)
        print("OCR init:", time.time() - t0)
    ocr = _ocr_cache[lang]



    arr = np.array(img.convert("RGB"))
    texts: list[str] = []
    try:
        t1=time.time()
        # PaddleOCR 3.x: predict() 返回结果对象, 用下标取 rec_texts
        results = ocr.predict(arr)
        print("OCR predict:", time.time() - t1)
        for res in results:
            rec = None
            try:
                rec = res["rec_texts"]          # 3.x 结果对象支持下标
            except (KeyError, TypeError, IndexError):
                rec = getattr(res, "rec_texts", None)  # 兜底: 当属性取
            if rec:
                texts.extend(rec)
    except AttributeError:
        # 兜底: PaddleOCR 2.x 旧 API
        result = ocr.ocr(arr)
        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])

    return " ".join(texts)


def compute_text_errors(sr: Image.Image, hr: Image.Image, lang: str = "en") -> dict:
    """文字一致性: CER / WER  方向: ↓ 越小越好.

    怎么算: 分别对 HR(真值, 当参考)和 SR 跑 OCR 得到文字, 用 jiwer 比较:
    CER = 字符错误率(增/删/改的字符数 / 参考字符总数)
    WER = 词错误率(同理, 以词为单位)
    原理: 这是你"防文字幻觉"最直接的量化——超分把 "3" 改成 "8"、
    把字猜错, CER/WER 立刻升高。HR 里没文字时返回 None(无可比)。
    """
    import jiwer
    if sr.size!=hr.size:
        sr=sr.resize(hr.size,Image.BICUBIC)
    ref = _ocr_text(hr, lang)
    hyp = _ocr_text(sr, lang)
    if not ref.strip():
        # 真值里没识别到文字, 无法计算错误率
        return {"cer": None, "wer": None, "ref_text": ref, "hyp_text": hyp}
    return {
          "cer": float(jiwer.cer(ref, hyp)),
          "wer": float(jiwer.wer(ref, hyp)),
          "ref_text": ref,
          "hyp_text": hyp,
    }


def compute_text_hallucination(cer: float | None, threshold: float = 0.1) -> float | None:
      """文字幻觉标志 (自定义)  方向: ↓ 越小越好, 取值 0 或 1.

      怎么算: 若该图的 CER 超过阈值(默认 0.1=10%字符变了), 判定这张图发生了
              文字幻觉, 记 1; 否则记 0。CER 为 None(无文字)时返回 None。
      原理: 把连续的 CER 变成一个"是否出错"的硬判定。
            注意: 单张图是 0/1; 你说的"幻觉率(Rate)"是在整个数据集上对这些
            0/1 求平均得到的, 在评测脚本里聚合, 不在这里。
      """
      if cer is None:
          return None
      return 1.0 if cer > threshold else 0.0



# ============================================================
# 总入口
# ============================================================
def evaluate(
    sr_output: Image.Image,
    ground_truth: Image.Image | None = None,
    *,
    run_text: bool = True,
    ocr_lang: str = "en",
    cer_threshold: float = 0.3,
) -> dict:
    """评估一张 SR 输出的质量, 返回所有指标的字典.

    Args:
          sr_output: 超分输出图。
          ground_truth: 高分辨率真值(HR)。为 None 时只算无参考指标(MANIQA/MUSIQ)。
          run_text: 是否跑 OCR 文字指标(CER/WER/文字幻觉)。慢, 可关。
          ocr_lang: OCR 语言, "en" 或 "ch"。
          cer_threshold: 判定文字幻觉的 CER 阈值。

    Returns:
          dict: 指标名 -> 值。算不了的指标为 None。方向见 METRIC_DIRECTION。
    """
    result: dict = {}

    # --- 无参考: 部署也能用 ---
    result["maniqa"] = compute_maniqa(sr_output)
    result["musiq"] = compute_musiq(sr_output)

    if ground_truth is None:
        # 没真值, 全参考/语义指标都算不了
        for k in ["psnr", "ssim", "lpips", "dists", "cer", "wer",
                 "text_halluc"]:
            result[k] = None
        return result

      # SR 与 HR 尺寸对齐 (全参考指标要求同尺寸)
    if sr_output.size != ground_truth.size:
        sr_aligned = sr_output.resize(ground_truth.size, Image.BICUBIC)
    else:
        sr_aligned = sr_output

    # --- 保真度 ---
    result["psnr"] = compute_psnr(sr_aligned, ground_truth)
    result["ssim"] = compute_ssim(sr_aligned, ground_truth)

    # --- 感知 (全参考) ---
    result["lpips"] = compute_lpips(sr_aligned, ground_truth)
    result["dists"] = compute_dists(sr_aligned, ground_truth)

    
    # --- 语义: 文字 ---
    if run_text:
        te = compute_text_errors(sr_aligned, ground_truth, lang=ocr_lang)
        result["cer"] = te["cer"]
        result["wer"] = te["wer"]
        result["text_halluc"] = compute_text_hallucination(te["cer"], cer_threshold)
    else:
        result["cer"] = result["wer"] = result["text_halluc"] = None

    return result

