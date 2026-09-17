EF2FS — Embedded feature fusion for multi-view multi-label feature selection
=============================================================================

题录
----
作者   : Pingting Hao, Wanfu Gao, Liang Hu
期刊   : Pattern Recognition, Volume 157 (2025)
文章号 : 110888
日期   : 2025-01（在线 2024-08-13）
DOI    : 10.1016/j.patcog.2024.110888
链接   : https://doi.org/10.1016/j.patcog.2024.110888

全文位置
--------
pre-pdf/EF2FS-2025-Pattern-Recognition-Embedded-feature-fusion.pdf
（11 页，2.40 MB；`pdfinfo` 的 Subject 字段直接给出
"Pattern Recognition, 157 (2025) 110888. doi:10.1016/j.patcog.2024.110888"，与题录一致）

对应代码
--------
alg/EF2FS.py  →  入口函数 EF2FS()（比其他算法多一个 `V_dim`：隐空间维数）
本仓库六个可运行算法之一，已通过 `bash run.sh --check` 验证。

> ⚠️ 该模块在上游版本中因 `Lx_lst` 未定义而完全无法运行（第一次迭代即 `NameError`），
> 本仓库已修复，详见 `FIXES.md` 第 3 节。

论文目标函数（EF2FS_ver2）
--------------------------
```
min_{A,G,B,w⁽ⁱ⁾}  Σᵢ ‖x⁽ⁱ⁾w⁽ⁱ⁾ - G‖²_F      （各视图向公共嵌入对齐）
                + α ‖Y - G Bᵀ‖²_F            （标签拟合）
                + β ‖X_f A - G‖²_F           （融合特征重构嵌入）
                + γ Σᵢ ‖a⁽ⁱ⁾ - w⁽ⁱ⁾‖²_F      （嵌入系数与视图投影一致）
                + δ ‖W‖₂,₁                   （行稀疏 → 特征排序）
```
其中 `X_f` 为视图加权融合特征，`G` 为公共低维嵌入，`A` 为特征系数矩阵，
`B` 建立嵌入到标签的映射，`w⁽ⁱ⁾` 为各视图投影，`a⁽ⁱ⁾` 为 `A` 按视图切分的块。

论文符号 ↔ 代码变量
-------------------
| 论文符号 | 代码变量 | 说明 |
|---|---|---|
| `G` | `V` | 公共低维嵌入（维数 `V_dim`）**代码里叫 `V`** |
| `X_f` | `new_X` | 视图加权融合特征（`nu[i]` 加权后拼接） |
| `A` | `A` | 特征系数矩阵（排序依据） |
| `B` | `B` | 嵌入 → 标签映射 |
| `w⁽ⁱ⁾` | `w[i]` | 各视图投影 |
| `a⁽ⁱ⁾` | `A[t1:t1+m[i], :]` | `A` 的视图切分块 |
| 视图权重 `v⁽ⁱ⁾` | `nu[i]` | 由 `1/tr(GᵀL⁽ⁱ⁾G)` 归一化 |
| 各视图图拉普拉斯 `L⁽ⁱ⁾` | `Lx_lst[i]` | `construct_W`（knn k=5，热核）构建 |
| `α, β, γ, δ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> 注意：代码中用 `V` 表示论文的嵌入 `G`，容易与其它算法里的 `V` 混淆；
> 阅读 `EF2FS.py` 时请以本表为准。
