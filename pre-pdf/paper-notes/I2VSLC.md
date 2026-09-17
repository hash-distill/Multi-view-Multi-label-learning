I2VSLC — Exploring view-specific label relationships for multi-view multi-label feature selection
=================================================================================================

题录
----
作者   : Pingting Hao, Weiping Ding, Wanfu Gao, Jialong He
期刊   : Information Sciences, Volume 681 (2024)
文章号 : 121215
日期   : 2024-10
DOI    : 10.1016/j.ins.2024.121215
链接   : https://doi.org/10.1016/j.ins.2024.121215

全文位置
--------
pre-pdf/I2VSLC-2024-Information-Sciences-View-specific-label-relationships.pdf
（14 页，2.15 MB；`pdfinfo` 的 Subject 字段给出
"Information Sciences, 681 (2024) 121215. doi:10.1016/j.ins.2024.121215"，与题录一致）

对应代码
--------
alg/I2VSLC.py  →  入口函数 I2VSLC()
本仓库六个可运行算法之一，已通过 `bash run.sh --check` 验证。

核心思想
--------
多数方法假设所有视图共享**同一个**标签相关性矩阵；但不同视图（如图像特征与文本
特征）反映出的标签共现关系并不相同，强行共享会削弱各视图自身的判别力。
I2VSLC 为**每个视图单独建标签关系图**，在标签空间做流形平滑，同时保留视图间共识。

论文中已验证的关键公式（式 6–7）
--------------------------------
视图特有标签的自刻画函数（self-portrait of the specific label function）：

```
Φ(·) = Σᵢ Tr(y⁽ⁱ⁾ᵀ L⁽ⁱ⁾ y⁽ⁱ⁾)  +  Σᵢ Σ_{j≠i} ‖y⁽ⁱ⁾ - y⁽ʲ⁾‖²_F
```

即「视图内标签图平滑」+「视图间标签共识」两项。这与代码完全对应：

- `Tr(y⁽ⁱ⁾ᵀ L⁽ⁱ⁾ y⁽ⁱ⁾)` → 代码主循环里的 `alpha*np.dot(Sx_lst[i], y[i])`
  与分母 `alpha*np.dot(Ax_lst[i], y[i])`
- `Σᵢ Σ_{j≠i} ‖y⁽ⁱ⁾ - y⁽ʲ⁾‖²_F` → 代码里用 `sum_Yt`（各视图伪标签的并）
  与 `tem1 = sum_Yt - y[i]` 实现，即把"其余视图的并"作为共识目标

此外论文明确指出低层特征选择使用 **`l₂,₁` 范数**（式 12 附近）度量特征行的重要性，
对应代码中由 `gamma`、`lamb` 两个加权对角阵 `D`、`C` 施加的行稀疏约束。

论文符号 ↔ 代码变量
-------------------
| 论文符号 | 代码变量 | 说明 |
|---|---|---|
| `y⁽ⁱ⁾` | `y[i]` | 各视图伪标签 |
| `L⁽ⁱ⁾` | `Lx_lst[i] = Ax_lst[i] - Sx_lst[i]` | **标签图**拉普拉斯（knn k=5，热核） |
| `Σᵢ y⁽ⁱ⁾` 二值化 | `sum_Yt` | 视图共识 |
| `AA` | `AA` | 共享标签相关矩阵，`Y·AA` 得视图无关的标签关联 |
| `w⁽ⁱ⁾` | `w[i]`，拼接为 `B` | 特征权重（排序依据：`B` 的逐行 2-范数） |
| `α, β, γ, λ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> 注意：论文中的标签拉普拉斯记作 `L⁽ⁱ⁾`（下标为标签），代码里对应 `Lx_lst[i]`；
> 而 `x[i]` 表示特征块，不要与 `Lx_lst` 的前缀混淆。

> 该模块在 `temp2` 出现 `NaN` 时会打印 `temp2-nan` 并 `break`，
> 属原始实现自带的保护逻辑，保留未改动。
