GRAFS — Anchor-guided global view reconstruction for multi-view multi-label feature selection
==============================================================================================

题录
----
作者   : Pingting Hao, Kunpeng Liu, Wanfu Gao
期刊   : Information Sciences, Volume 679 (2024)
文章号 : 121124
日期   : 2024-09（在线 2024-07-02）
DOI    : 10.1016/j.ins.2024.121124
链接   : https://doi.org/10.1016/j.ins.2024.121124

全文位置
--------
pre-pdf/GRAFS-2024-Information-Sciences-Anchor-guided-global-view-reconstruction.pdf
（13 页，1.47 MB；`pdfinfo` 的 Subject 字段给出
"Information Sciences, 679 (2024) 121124. doi:10.1016/j.ins.2024.121124"，与题录一致）

对应代码
--------
alg/GRAFS.py  →  入口函数 view6()（`kk` 为锚点数）
本仓库六个可运行算法之一，已通过 `bash run.sh --check` 验证。

核心思想
--------
直接对全样本建模多视图关系，复杂度随样本数平方增长。GRAFS 引入**锚点**把视图重构
压缩到紧凑空间，同时联合学习「全局视图表示」与「特征权重」，并用标签侧的语义信息
引导这一重构过程。

论文中已验证的关键公式（式 6–8）
--------------------------------
候选视图提取（candidate view extraction）：

```
       Σᵢ vᵢ ‖X⁽ⁱ⁾ - B P⁽ⁱ⁾‖²_F          （用锚点重构各视图）
L_E =  + ‖B - W_c A_c R_c‖²_F             （锚点侧多层分解）
       + ‖X_f - W_c A_f R_f‖²_F           （全局视图侧多层分解）
       + ‖A_c - A_f‖²_F                   （两条分解路径相互一致）
```

这里 `X_f` 是**重构出的全局视图**（不是原特征拼接），`B` 是锚点矩阵，`P⁽ⁱ⁾` 是
视图到锚点的映射，`vᵢ` 是视图权重，`A_c/A_f` 与 `W_c` 刻画共享结构。

论文符号 ↔ 代码变量
-------------------
| 论文符号 | 代码变量 | 说明 |
|---|---|---|
| `B` | `B` | 锚点矩阵（列数 `kk`） |
| `X_f` | `X1` | 全局视图表示 |
| `P⁽ⁱ⁾` | `p[i]` | 视图 → 锚点映射 |
| `W_c` | `W1` | 共享权重 |
| `A_c`、`A_f` | `A1`、`A2` | 两条分解路径的结构矩阵 |
| `R_c`、`R_f` | `R1`、`R2` | 对应系数矩阵 |
| `L_E` 中的 `vᵢ` | `nu[i]` | 视图权重（由 `1/tr(x⁽ⁱ⁾ᵀL_y x⁽ⁱ⁾)` 归一化） |
| `L_y` | `Ly` | **标签图**拉普拉斯（knn k=20，热核） |
| `W` | `W` | 全局特征权重（排序依据） |
| `X⁽ⁱ⁾` | `x[i]`；选择矩阵 | `s[i]`（视图侧）、`d[i]`（全局侧） |
| `α, β, γ, λ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> 提示：该模块内部 `record['method']` 原写作 `'view6'`，本仓库已改为 `'GRAFS'`
> 以便输出可读（仅为记录字符串，不参与计算）。
