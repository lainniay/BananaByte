
可以。综合前面的讨论，我建议你把最终方案收敛成一个更干净的框架：

> **通用生成式模型只负责产生 RGB 自然候选图；物理/频率模块负责决定哪里、以什么频率、以多大权重采纳生成结果。**

不要再让生成式模型直接输出 HSV、深度、透射率、小波系数或傅立叶谱。

---

# 最终可行方案

可以命名为：

**PCGF-UIR: Physics-Constrained Generative Fusion for Underwater Image Restoration**

中文：

**物理约束的生成式水下图像修复融合方法**

核心结构：

[
I
\rightarrow
{J_{\text{phy}}, J_{\text{gen}}}
\rightarrow
W(x)
\rightarrow
\text{Lab + Wavelet Fusion}
\rightarrow
J_{\text{out}}
]

其中：

* (I)：原始水下图像；
* (J_{\text{phy}})：保守物理恢复图；
* (J_{\text{gen}})：GPT Image / Nano Banana 生成的 RGB 恢复图；
* (W(x))：生成图可信度权重图；
* (J_{\text{out}})：最终结果。

---

# 1. 保留的模块

最终只保留四个核心模块。

## 模块一：保守物理恢复分支

先从原图得到一个保守恢复结果：

[
J_{\text{phy}}=F_{\text{phy}}(I)
]

这个分支不需要做得很复杂。建议使用：

1. 白平衡；
2. 红通道补偿；
3. 轻量去雾；
4. CLAHE 或局部对比度增强。

它的作用不是取得最佳视觉效果，而是提供一个**结构可信、幻觉少、物理方向正确**的基准图。

可以理解为：

> (J_{\text{phy}}) 负责保真，(J_{\text{gen}}) 负责自然性。

---

## 模块二：生成式 RGB 候选分支

输入原始图或者轻度预处理后的图，得到：

[
J_{\text{gen}}=G(I)
]

这里生成模型只输出 RGB 图像，不输出任何中间物理域。

prompt 应该强调保守修复：

```text
Restore this underwater image into a natural and physically plausible appearance.
Remove underwater color cast and haze.
Recover plausible object colors.
Preserve all geometry, object boundaries, textures, and camera viewpoint.
Do not add, remove, or move any object.
Do not stylize.
Do not over-brighten the foreground subject.
```

如果计算预算允许，可以生成 3 个候选：

[
{J_{\text{gen}}^1,J_{\text{gen}}^2,J_{\text{gen}}^3}
]

然后选择结构最稳定的一个。预算有限时，直接用一个候选也可以。

---

## 模块三：生成图可信度估计

不要用全局比例融合，而是估计一个像素级权重：

[
W(x)\in[0,1]
]

它表示当前位置可以相信生成图的程度。

最终融合形式是：

[
J_{\text{out}}(x)
=================

W(x)J_{\text{gen}}(x)
+
(1-W(x))J_{\text{phy}}(x)
]

但这个融合不要直接在 RGB 上做，而是在 Lab + 小波域中做。

---

## 模块四：Lab + 小波多尺度融合

将 (J_{\text{phy}}) 和 (J_{\text{gen}}) 转到 Lab 空间：

[
J^{Lab}=(L,a,b)
]

然后对每个通道做小波分解：

[
L,a,b
\rightarrow
{LL,LH,HL,HH}
]

融合原则：

> **低频颜色更多相信生成图，高频结构更多相信物理图。**

具体策略：

| 通道       |   LL 低频 | LH/HL 中高频 |   HH 高频 |
| -------- | ------: | --------: | ------: |
| (L) 亮度   | 少量使用生成图 |   基本不用生成图 |   不用生成图 |
| (a,b) 色度 | 较多使用生成图 |   少量使用生成图 | 基本不用生成图 |

一个简单初始权重可以是：

| 子带 | (L) 通道生成权重 | (a,b) 通道生成权重 |
| -- | ---------: | -----------: |
| LL |  (0.2W(x)) |    (0.7W(x)) |
| LH | (0.05W(x)) |    (0.2W(x)) |
| HL | (0.05W(x)) |    (0.2W(x)) |
| HH |        (0) |   (0.05W(x)) |

这样生成模型主要影响：

* 颜色；
* 大尺度色调；
* 自然外观。

但不轻易改变：

* 物体边缘；
* 高频纹理；
* 几何结构；
* 小目标。

---

# 2. 权重图 (W(x)) 怎么构造

建议用三个量构造，足够简洁。

---

## 2.1 退化程度图 (D(x))

退化越严重，越需要生成模型。

可以用红通道衰减、局部对比度和雾化程度估计：

[
D(x)
====

\lambda_r D_r(x)
+
\lambda_c D_c(x)
+
\lambda_h D_h(x)
]

其中：

[
D_r(x)=\max(0,G(x)+B(x)-2R(x))
]

表示红通道缺失程度。

[
D_c(x)=1-\text{LocalContrast}(I)(x)
]

表示局部对比度缺失。

[
D_h(x)
]

表示局部雾化或散射强度，可以用暗通道、饱和度下降或局部方差近似。

---

## 2.2 结构一致性图 (C(x))

如果生成图和物理图的边缘差异很大，说明生成模型可能幻觉。

定义：

[
C(x)
====

\exp
\left(
------

\frac{
|\nabla Y_{\text{gen}}(x)-\nabla Y_{\text{phy}}(x)|
}{\tau}
\right)
]

其中：

* (Y_{\text{gen}})：生成图亮度；
* (Y_{\text{phy}})：物理图亮度；
* (C(x)) 越大，说明结构越一致。

如果生成图新增了纹理、改变了轮廓，(C(x)) 会变小。

---

## 2.3 边缘保护图 (E(x))

在强边缘附近，应该减少生成图权重。

[
E(x)=\text{Normalize}(|\nabla Y_I(x)|)
]

边缘越强，生成权重越低。

---

## 最终权重

可以写成：

[
W(x)
====

\text{clip}
\left(
D(x)\cdot C(x)\cdot (1-E(x)),
0,1
\right)
]

再用 guided filter 或 bilateral filter 平滑：

[
W'(x)=\text{GuidedFilter}(W(x),Y_I)
]

这样得到的 (W'(x)) 有三个特性：

1. 退化严重处更相信生成图；
2. 结构不一致处不相信生成图；
3. 物体边界处保护原始结构。

---

# 3. 最终算法流程

完整流程如下。

```text
Input: underwater image I

1. Conservative physical restoration
   J_phy = WhiteBalance + RedCompensation + Dehaze + CLAHE

2. Generative RGB restoration
   J_gen = GPT Image / Nano Banana restoration

3. Compute degradation map
   D = red attenuation + low contrast + haze estimation

4. Compute structural consistency map
   C = exp(-|grad(Y_gen) - grad(Y_phy)| / tau)

5. Compute edge protection map
   E = normalize(|grad(Y_I)|)

6. Compute generation reliability map
   W = clip(D * C * (1 - E), 0, 1)
   W = GuidedFilter(W, Y_I)

7. Convert J_phy and J_gen to Lab

8. Wavelet decomposition
   Decompose L, a, b channels into LL, LH, HL, HH

9. Frequency-aware fusion
   L channel:
      low-frequency: weakly use J_gen
      high-frequency: mostly use J_phy

   a,b channels:
      low-frequency: strongly use J_gen
      high-frequency: weakly use J_gen

10. Inverse wavelet transform

11. Convert Lab back to RGB

Output: restored image J_out
```

---
