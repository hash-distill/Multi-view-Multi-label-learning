# 论文与代码对照简介

本文档简要介绍本仓库实现的**六篇论文**及其对应的代码模块，说明每篇论文要解决的问题、
核心模型思路、代码在哪、代码里哪些变量对应论文里的哪个符号。

> 说明：英文 `README.md` 与 `FIXES.md` 保持各自用途不变，本文档是新增的对照材料。
> `alg/` 下共有 **6 个可运行算法模块**，与 **6 篇论文**一一对应（第 1–6 节）。
> 另有 `THBFS`：仓库中仅有其一页宣传材料（现已收录于 `pre-pdf/`），
> **既无公开全文、也无实现代码**，见第 7 节。

## 全文收录状态

**六篇已发表论文的全文现已全部收录在 `pre-pdf/`**（详细清单与核验方式见
[`pre-pdf/README.md`](pre-pdf/README.md)）：

| 算法 | 全文 | 文件 |
|---|---|---|
| TOCL | ✅ 10 页 | `pre-pdf/TOCL-2025-ACM-MM-Tensor-based-Opposing-yet-Complementary-Learning.pdf` |
| UGRFS | ✅ 9 页 | `pre-pdf/AAAI-2025-UGRFS-Uncertainty-Aware-Global-View-Reconstruction.pdf`（另有原有作者版） |
| EF2FS | ✅ 11 页 | `pre-pdf/EF2FS-2025-Pattern-Recognition-Embedded-feature-fusion.pdf` |
| DHLI | ✅ 9 页 | `pre-pdf/AAAI-2024-DHLI-Double-Layer-Hybrid-Label-Identification.pdf` |
| I2VSLC | ✅ 14 页 | `pre-pdf/I2VSLC-2024-Information-Sciences-View-specific-label-relationships.pdf` |
| GRAFS | ✅ 13 页 | `pre-pdf/GRAFS-2024-Information-Sciences-Anchor-guided-global-view-reconstruction.pdf` |
| THBFS | ⚠️ 1 页宣传材料 | `pre-pdf/THBFS.pdf`（无全文、无代码） |

每份 PDF 的题名、作者、页数均与 Crossref 官方题录逐项核对通过，首页页眉的
DOI 也与题录一致。各篇的详细题录、摘要与「论文符号 ↔ 代码变量」对照见
`pre-pdf/paper-notes/`。

本文档中的模型描述**以论文正文与代码实现双向核对为准**：目标函数取自各篇原文的
公式编号，变量对应关系逐项核对了论文符号与 `alg/*.py` 中的实际变量名。

## 速查表

| # | 论文简称 | 发表处 | 代码文件 | 入口函数 | 代码里记录的 `method` |
|---|---|---|---|---|---|
| 1 | **TOCL** | ACM MM 2025 | `alg/TOCL.py` | `view7` | `TOCL` |
| 2 | **UGRFS** | AAAI 2025 | `alg/UGRFS.py` | `UGRFS` | `UGRFS` |
| 3 | **EF2FS** | Pattern Recognition 2025 | `alg/EF2FS.py` | `EF2FS` | `EF2FS` |
| 4 | **DHLI** | AAAI 2024 | `alg/DHLI.py` | `DHLI` | `DHLI` |
| 5 | **I2VSLC** | Information Sciences 2024 (681) | `alg/I2VSLC.py` | `I2VSLC` | `I2VSLC` |
| 6 | **GRAFS** | Information Sciences 2024 (679) | `alg/GRAFS.py` | `view6` | `GRAFS` |
| – | **THBFS** | 宣传材料（未公开全文） | 无 | – | – |

所有模块都遵守同一套接口契约（详见 `README_zh.md` 第 3 节）：

```python
record, n_iter = entry(X, view_dims, Y, dataset, alpha, beta, gamma, lamb [, V_dim | kk])
record['idx']          # 全部特征索引，按重要性降序
```

因此运行方式完全统一：

```bash
bash run.sh --alg TOCL  --data emotions
bash run.sh --alg UGRFS --data emotions
```

---

## 1. TOCL — 张量对立互补学习

**论文**：*Tensor-based Opposing yet Complementary Learning for Multi-view
Multi-label Feature Selection*，ACM MM 2025（第 33 届 ACM 国际多媒体会议）。
[ACM DL](https://dl.acm.org/doi/abs/10.1145/3746027.3755447)

**代码**：`alg/TOCL.py`，入口 `view7(X, x_view, Y, dataset, alpha, beta, gamma, lamb)`

### 要解决的问题

多视图多标签特征选择中，各视图既共享一致语义、又各自携带互补信息。逐视图独立建模
会丢失跨视图的高阶一致性；简单拼接又会引入冗余。TOCL 的思路是把各视图的两类标签
表示堆成一个**三阶张量**，用**张量核范数**约束其低秩性来提取跨视图共识。

### 建模要点

- 每个视图维护一份伪标签矩阵 `y[i]`，用其样本相似度构成视图关联矩阵
  `HH[:, :, i] = y[i] @ y[i].T`，多个视图堆叠成三阶张量。
- 对张量施加**低秩约束**，通过 `prox_weight_tensor_nuclear_norm()` 求解近端算子：
  沿第 3 维做 FFT → 对每个前切片做 SVD → 平方根阈值收缩 → 逆 FFT。
- 用一个正交矩阵 `P` 与逻辑回归式非线性映射 `NL = 1/(1+exp(-XV))` 分别刻画
  "视图融合"与"标签判别"，通过 `U`、`V` 分解把特征与标签语义对齐。
- 特征排序量：全局权重矩阵 `W` 的**逐行 2-范数**。

### 代码结构（`TOCL.py`）

| 位置 | 内容 |
|---|---|
| `prox_weight_tensor_nuclear_norm(Y, C)` | 张量核范数近端算子（本模块最耗时的部分） |
| `view7()` 第 1.1 步 | 更新各视图伪标签 `y[i]` + MinMax 归一化二值化 |
| `view7()` 第 1.2 步 | 张量低秩投影得到 `Z` |
| `view7()` 第 1.3–1.7 步 | 交替更新 `W`、`P`、`Y_n`、`U`、`V` |
| 目标函数 | `delta*temp1 + aaa*temp3 + alpha*(temp4+temp5) + 2*beta*temp6 + gamma*(temp7+temp8) + 2*lamb*temp9` |
| 排序 | `w_2 = LA.norm(W, ord=2, axis=1)`，`argsort(-w_2)` |

### 变量对照

| 论文符号 | 代码变量 |
|---|---|
| 视图伪标签 | `y[i]` |
| 视图关联张量 | `HH`（原张量） / `Z`（低秩投影后） |
| 全局特征权重 | `W` |
| 视图融合矩阵 | `P` |
| 标签判别分解 | `U`、`V`（秩 `kk = int(0.8 * 标签数)`） |
| 标签不一致项 | `Y_n` |
| 超参数 | `alpha, beta, gamma, lamb`；`delta=aaa=rho=1` 在函数内写死 |

### 运行特征

**本仓库最慢的算法**：每轮迭代要做一次「FFT + 逐切片 SVD + 逆 FFT」，
在 `emotions`（593 样本）上单折约 11 秒（60 轮）。建议用
`MVML_MAX_ITER` 限制迭代预算：

```bash
MVML_MAX_ITER=60 bash run.sh --alg TOCL --data emotions
```

---

## 2. UGRFS — 不确定度感知的全局视图重构

**论文**：*Uncertainty-Aware Global-View Reconstruction for Multi-View
Multi-Label Feature Selection*，AAAI 2025。
全文 PDF 已随仓库提供：`pre-pdf/25-aaai-Uncertainty-Aware Global-View Reconstruction....pdf`

**代码**：`alg/UGRFS.py`，入口 `UGRFS(X, x_view, Y, dataset, alpha, beta, gamma, lamb)`

### 要解决的问题

已有方法通常把「一致性部分」和「互补部分」分开提取，分割边界不清会引入噪声；
同时它们默认**每个样本的可信度都等于 1**，而真实数据里样本质量参差不齐。
此外，直接拼接多视图会忽略视图之间的关系。

### 建模要点

论文把模型写成全局视图重构与不确定度感知特征选择之和，`L = L_G(·) + L_U(·)`：

1. **视图关系融合**：用各视图局部几何结构相似度估计视图贡献
   `X_f = [ν₁X⁽¹⁾, ν₂X⁽²⁾, …, ν_V X⁽ⱽ⁾]`，各视图权重 `ν_i` 由
   `1/tr(X⁽ⁱ⁾ᵀ L⁽ⁱ⁾ X⁽ⁱ⁾)` 归一化得到（与 `GRAFS`/`EF2FS` 同一思路）。
2. **全局视图分布**：对标签矩阵做高斯核非线性映射
   `ρ(Y) = exp(-(J₁Y - YYᵀ)² / avg(pdist)²)`，再用系数矩阵 `Ŵ_y` 回归
   `D = ρ(Y)Ŵ_y + b = Y_x W_y`，其中 `Y_x = [ρ(Y), 1]`。
3. **重构 + 结构保持**：
   `L_G = ‖D - X_f‖²_F + Tr(Dᵀ L_Y D)`。
4. **样本不确定度**：为每个视图、每个样本学习置信度矩阵 `C⁽ⁱ⁾`，把重构拆解到各视图：
   `L_U = Σ_i ‖diag(C⁽ⁱ⁾)X⁽ⁱ⁾W⁽ⁱ⁾ - Y‖²_F + Σ_i ‖D - diag(C⁽ⁱ⁾)X⁽ⁱ⁾‖²_F + ‖W‖₂,₁`。
5. 用 `‖W‖₂,₁ = 2Tr(WᵀEW)`（`e_ii = 1/(2‖W_i‖₂)`）做行稀疏，最终按
   `‖W_(j)‖₂` 排序特征。

### 代码结构（`UGRFS.py`）

| 位置 | 内容 |
|---|---|
| `kernelmatrix(par, trainX, testX)` | 高斯核，用于构造 `ρ(Y)`；`par = 1/mean(pdist(Y))` |
| `UGRFS()` 初始化段 | 各视图权重 `nu`、`new_X`（即 `X_f`）、`Ly`、`Yx`、`wy` |
| 主循环第 1 段 | 更新 `w[i]`（含 `lamb` 的 `‖·‖₂,₁` 加权对角阵 `D`） |
| 主循环第 2 段 | 更新样本置信度 `c[i]` |
| 主循环第 3 段 | 更新 `wy[i]`（全局视图分布系数） |
| 目标函数 | `temp4 + alpha*temp2 + beta*temp3 + gamma*temp1 + lamb*temp5` |
| 排序 | `LA.norm(W, ord=2, axis=1)`，`argsort(-w_2)` |

### 变量对照

| 论文符号 | 代码变量 | 含义 |
|---|---|---|
| `X_f` | `new_X` | 视图加权融合矩阵 |
| `ν_i` | `nu[i]` | 视图权重 |
| `ρ(Y)`、`Y_x` | `Yx` | 标签核映射与增广 |
| `D` | `p2 = Yx @ Wy` 或逐视图 `p1 = c[i] @ x[i]` | 全局视图分布 |
| `W_y`（按视图拆） | `wy[i]` | 拆分后的系数矩阵 |
| `C⁽ⁱ⁾` | `c[i]`（对角矩阵） | 样本置信度 |
| `L_Y` | `Ly` | 标签图拉普拉斯（knn k=20，热核） |
| `W`、`W⁽ⁱ⁾` | `W`（各视图 `w[i]` 拼接） | 特征权重 |
| `α, β, γ, δ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> ⚠️ 论文里的第 4 个平衡参数是 `δ`，代码中对应位置用的是 `lamb`；
> 另外 `δ, α, β, γ` 在论文里是四个独立超参，而代码统一按
> `alpha, beta, gamma, lamb` 接收（默认均为 1.0）。

### 论文报告的基准结果（供参照）

论文在 6 个数据集上评测（yeast / SCENE / VOC07 / MIRFlickr / IAPRTC12 / 3Sources），
表 2 给出的 AP 为：yeast 0.6725、SCENE 0.8010、VOC07 0.5871、MIRFlickr 0.6770、
IAPRTC12 0.1474、3Sources 0.4728（均为 5 次均值 ± 标准差，优于 M2LD / MSFS /
MoRE / MRDM / MIFS / CLML 等对比方法）。

---

## 3. EF2FS — 嵌入式特征融合

**论文**：*Embedded feature fusion for multi-view multi-label feature selection*，
Pattern Recognition 157 (2025) 110888。
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320324006393)

**代码**：`alg/EF2FS.py`，入口
`EF2FS(X, x_view, Y, dataset, alpha, beta, gamma, lamb, V_dim)`
（比其他算法多一个 `V_dim`：隐空间维数）

### 要解决的问题

"先融合视图、再做特征选择"的两阶段做法会让融合目标与选择目标不一致。
EF2FS 把**视图融合与特征选择嵌入同一个优化框架**，让两者互相指导。

### 建模要点

论文的目标函数（原文记为 `EF2FS_ver2`）为：

```
min_{A,G,B,w⁽ⁱ⁾}  Σᵢ ‖x⁽ⁱ⁾w⁽ⁱ⁾ - G‖²_F      （各视图向公共嵌入对齐）
                + α ‖Y - G Bᵀ‖²_F            （标签拟合）
                + β ‖X_f A - G‖²_F           （融合特征重构嵌入）
                + γ Σᵢ ‖a⁽ⁱ⁾ - w⁽ⁱ⁾‖²_F      （嵌入系数与视图投影一致）
                + δ ‖W‖₂,₁                   （行稀疏 → 特征排序）
```

- `G ∈ R^(n×V_dim)` 是共享的低维嵌入，各视图投影 `w⁽ⁱ⁾` 向它对齐。
- `B` 建立嵌入到标签的映射（`Y ≈ G Bᵀ`），把嵌入与标签语义绑定。
- `X_f` 是视图加权融合特征，系数矩阵 `A` 要求 `X_f A ≈ G`，
  即融合表示可解释为特征的线性组合。
- `a⁽ⁱ⁾` 是 `A` 按视图切分的块，与 `w⁽ⁱ⁾` 一致；`A` 受 `‖·‖₂,₁` 稀疏约束，
  逐行 2-范数即为特征重要性。
- 视图权重 `vᵢ = (1/tr(GᵀL⁽ⁱ⁾G))` 归一化，`L⁽ⁱ⁾` 为各视图图拉普拉斯。

### 代码结构（`EF2FS.py`）

| 位置 | 内容 |
|---|---|
| 循环外 | 用 `construct_W`（knn k=5，热核）为每个视图构建 `Lx_lst` |
| 第 1 步 | 计算视图权重 `nu`，拼出 `new_X` |
| 第 2 步 | 交替更新 `B` → `V` → `A` → `w[i]` |
| 目标函数 | `temp1 + alpha*temp2 + beta*temp3 + gamma*temp4 + lamb*temp5` |
| 排序 | `LA.norm(A, ord=2, axis=1)`，`argsort(-w_2)` |

目标函数各项含义（按代码中的 `temp` 编号）：`temp1 = ‖X_f·A - G‖²_F`（融合重构）、
`temp2 = ‖Y - G·Bᵀ‖²_F`（标签拟合）、
`temp3 = Σᵢ ‖x⁽ⁱ⁾·w⁽ⁱ⁾ - G‖²_F`（视图对齐）、
`temp4 = Σᵢ ‖a⁽ⁱ⁾ - w⁽ⁱ⁾‖²_F`（系数与投影一致）、
`temp5 = ‖A‖₂,₁`（行稀疏）。代码中 `V` 即论文的 `G`。

### 变量对照

| 论文符号 | 代码变量 | 含义 |
|---|---|---|
| `G` | `V` | 公共低维嵌入（维数 `V_dim`）— **代码里叫 `V`** |
| `X_f` | `new_X` | 视图加权融合特征（`nu[i]` 加权后拼接） |
| `A` | `A` | 特征系数矩阵（**排序依据**） |
| `B` | `B` | 嵌入 → 标签映射 |
| `w⁽ⁱ⁾` | `w[i]` | 各视图投影 |
| `a⁽ⁱ⁾` | `A[t1:t1+m[i], :]` | `A` 的视图切分块 |
| 视图权重 `vᵢ` | `nu[i]` | 由 `1/tr(GᵀL⁽ⁱ⁾G)` 归一化 |
| 各视图图拉普拉斯 `L⁽ⁱ⁾` | `Lx_lst[i]` | `construct_W`（knn k=5，热核）构建 |
| `α, β, γ, δ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> ⚠️ 注意两处命名差异：论文的第 4 个平衡参数是 `δ`，代码里叫 `lamb`；
> 论文的嵌入记作 `G`，而代码里复用了字母 `V`。阅读 `EF2FS.py` 时请以本表为准。

> 📌 **重要修复记录**：原仓库该文件在视图权重那一步引用了**从未定义的变量
> `Lx_lst`**，导致第一次迭代就 `NameError`，即 **EF2FS 在上游版本中根本无法运行**。
> 现已在循环前按与 `I2VSLC` 相同的方式补齐图拉普拉斯构建（并对拉普拉斯迹为 0
> 的退化情形加了保护）。详见 `FIXES.md` 第 3 节。

---

## 4. DHLI — 双层级混合标签识别

**论文**：*Double-Layer Hybrid-Label Identification Feature Selection for
Multi-View Multi-Label Learning*，AAAI 2024, 38(11): 12295-12303。
[AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/29120)

**代码**：`alg/DHLI.py`，入口 `DHLI(X, x_view, Y, dataset, alpha, beta, gamma, lamb)`

### 要解决的问题

多标签数据里，标签矩阵并非"完全正确"：既存在视图间共享的一致标签，也存在被噪声
污染的错误标签，还存在只属于某个视图的特有标签。直接拿观测标签做回归会把噪声也学进去。

### 建模要点（双层级）

DHLI 把观测标签矩阵分解为三部分，并在两层上交替识别：

- **第一层（标签层面）**：`Y_c`（一致 / 干净标签）、`Y_n`（噪声标签）、
  `Y_spe = Y - Y_c - Y_n`（视图特有标签），三者互相约束。
- **第二层（特征层面）**：每个视图同时学两组权重
  `w[i]`（拟合一致标签 `Y_c`）与 `u[i]`（拟合各视图自身伪标签 `y[i]`），
  用一个"视图间互斥"项耦合。
- 每个视图维护伪标签 `y[i]`，与"其他视图伪标签的并按"比较，形成投票式共识。
- 特征重要性 = 归一化后 `MinMax(W) + MinMax(U)` 的**逐行 2-范数**。

### 代码结构（`DHLI.py`）

| 位置 | 内容 |
|---|---|
| 初始化 | 各视图 `w[i]`、`u[i]`、伪标签 `y[i]`、`Y_c`、`Y_n` |
| 主循环第 1 段 | 更新 `w[i]`（拟合 `Y_c`，带 `‖·‖₂` 加权对角阵 `D`） |
| 主循环第 1 段 | 更新 `u[i]`（拟合 `y[i]`，带对角阵 `E`） |
| 主循环第 1 段 | 更新 `y[i]`（含视图互斥项 `lamb*y[i]*B*B`） |
| 主循环第 2 段 | MinMax 归一化 + 0.5 阈值二值化各 `y[i]` |
| 主循环第 3 段 | 更新 `Y_c`（带 `alpha*Y` 与 `lamb*tem2`） |
| 主循环第 4 段 | 更新 `Y_n`（带 `l2` 加权 `Q = 1/(2|Y_n|)`） |
| 目标函数 | `temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8` |
| 排序 | `A = MinMax(W) + MinMax(U)`；`LA.norm(A, ord=2, axis=1)` |

### 变量对照

| 论文符号 | 代码变量 | 含义 |
|---|---|---|
| `Y_c` | `Y_c` | 一致（干净）标签 |
| `Y_n` | `Y_n` | 噪声标签 |
| `Y_spe` | `Y_spe` | 视图特有标签 |
| `y⁽ⁱ⁾` | `y[i]` | 第 i 个视图的伪标签 |
| `W⁽ⁱ⁾`、`U⁽ⁱ⁾` | `w[i]`、`u[i]` | 两组视图特征权重 |
| 视图互斥乘积 | `B`（`= ∏_{j≠i} y[j]`） | 其他视图伪标签的 Hadamard 积 |
| `Σ_i y⁽ⁱ⁾` 二值化 | `sum_yt` | 视图共识 |
| `α, β, γ, λ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> 📌 **重要修复记录**：原代码把互斥乘积写成
> `B = 0` 后 `B = B * y[i]`，使 `B` **恒为 0**，导致 `lamb*y[i]*B*B` 这一
> 视图互斥正则项**一直静默失效**（且乘的是自己而非其他视图）。现已修正为
> `B = ∏_{j≠i} y[j]`。详见 `FIXES.md` 第 4 节。

---

## 5. I2VSLC — 视图特有标签关系

**论文**：*Exploring view-specific label relationships for multi-view
multi-label feature selection*，Information Sciences 681 (2024) 121215。
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025524011290)

**代码**：`alg/I2VSLC.py`，入口 `I2VSLC(X, x_view, Y, dataset, alpha, beta, gamma, lamb)`

### 要解决的问题

多数方法假设所有视图共享**同一个**标签相关性矩阵。但不同视图（如图像特征 vs 文本
特征）反映出的标签共现关系并不相同，强行共享会削弱各视图自身的判别力。

### 建模要点

论文为每个视图单独建**标签关系图**，并给出视图特有标签的自刻画函数（式 6–7）：

```
Φ(·) = Σᵢ Tr(y⁽ⁱ⁾ᵀ L⁽ⁱ⁾ y⁽ⁱ⁾)  +  Σᵢ Σ_{j≠i} ‖y⁽ⁱ⁾ - y⁽ʲ⁾‖²_F
```

- 第一项 `Tr(y⁽ⁱ⁾ᵀL⁽ⁱ⁾y⁽ⁱ⁾)` 在**标签空间**做流形平滑，`L⁽ⁱ⁾ = A⁽ⁱ⁾ - S⁽ⁱ⁾`
  是该视图的标签图拉普拉斯（knn k=5，热核）；这正对应代码里的
  `alpha*np.dot(Sx_lst[i], y[i])` 与分母中的 `alpha*np.dot(Ax_lst[i], y[i])`。
- 第二项 `ΣᵢΣ_{j≠i}‖y⁽ⁱ⁾-y⁽ʲ⁾‖²_F` 促进视图间标签共识，代码用
  `sum_Yt`（各视图伪标签的并）与 `tem1 = sum_Yt - y[i]` 实现，
  即把"其余视图的并"当作共识目标。
- 学一个共享标签相关矩阵 `AA`，用 `Y·AA` 得到与视图无关的标签关联表示，
  与共识对齐（对应 `beta`）。
- 论文明确指出低层特征选择使用 **`l₂,₁` 范数**度量特征行的重要性，
  对应代码中由 `gamma`、`lamb` 两个加权对角阵 `D`、`C` 施加的行稀疏约束。
- 排序量取各视图 `w[i]` 纵向拼接后的 `B` 的逐行 2-范数。

### 代码结构（`I2VSLC.py`）

| 位置 | 内容 |
|---|---|
| 循环前 | 各视图 `construct_W(v[i], k=5, heat_kernel)` → `Sx_lst` / `Ax_lst` / `Lx_lst` |
| 主循环 | 更新 `w[i]`（两个加权对角阵 `D`、`C`） |
| 主循环 | 更新共享标签相关矩阵 `AA` |
| 主循环 | 更新各视图伪标签 `y[i]`（含图平滑项与共识项） |
| 主循环 | MinMax 归一化 + 二值化 `y[i]`，更新 `sum_Yt` |
| 目标函数 | `temp1 + alpha*temp2 + beta*temp3 + gamma*temp4 + lamb*temp5` |
| 排序 | `B = concat(w)`；`LA.norm(B, ord=2, axis=1)` |

目标函数各项：`temp1 = Σ_i‖v[i]w[i] - y[i]‖²_F`（视图拟合）、
`temp2 = Σ_i Tr(y[i]ᵀL_x y[i]) + Σ_{i<j}‖y[i]-y[j]‖²_F`（视图内标签平滑 + 视图间共识）、
`temp3`（标签关联 `Y·AA` 与共识的两侧一致性）、
`temp4 = Σ_i‖w[i]‖_F`、`temp5 = ‖B‖₂,₁`。

### 变量对照

| 论文符号 | 代码变量 | 含义 |
|---|---|---|
| `L⁽ⁱ⁾` | `Lx_lst[i] = Ax_lst[i] - Sx_lst[i]` | 标签图拉普拉斯（由 `Sx_lst`/`Ax_lst` 相减得到） |
| `S⁽ⁱ⁾`、`A⁽ⁱ⁾` | `Sx_lst[i]`、`Ax_lst[i]` | 标签图亲和矩阵 / 度矩阵（knn k=5，热核） |
| `AA` | `AA` | 共享标签相关矩阵，`Y·AA` 得视图无关的标签关联 |
| `y⁽ⁱ⁾` | `y[i]` | 各视图伪标签 |
| `Σ_i y⁽ⁱ⁾` 二值化 | `sum_Yt` | 视图共识 |
| `w⁽ⁱ⁾` | `w[i]`，拼接为 `B` | 特征权重（排序依据：`B` 的逐行 2-范数） |
| `α, β, γ, λ` | `alpha, beta, gamma, lamb` | 平衡参数 |

> 注：该模块在 `temp2` 出现 `NaN` 时会打印 `temp2-nan` 并 `break`，
> 属于原始实现自带的保护逻辑，保留未改动。

---

## 6. GRAFS — 锚点引导的全局视图重构

**论文**：*Anchor-guided global view reconstruction for multi-view multi-label
feature selection*，Information Sciences 679 (2024) 121124。
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025524010387)

**代码**：`alg/GRAFS.py`，入口
`view6(X, x_view, Y, dataset, alpha, beta, gamma, lamb, kk)`
（`kk` 为锚点数）

### 要解决的问题

直接对全样本建模多视图关系，复杂度随样本数平方增长。GRAFS 引入**锚点**把视图
重构压缩到紧凑空间，同时联合学习"全局视图表示"与"特征权重"。

### 建模要点

论文中「候选视图提取」的目标函数（式 6–8）为：

```
       Σᵢ vᵢ ‖X⁽ⁱ⁾ - B P⁽ⁱ⁾‖²_F          （用锚点重构各视图）
L_E =  + ‖B - W_c A_c R_c‖²_F             （锚点侧多层分解）
       + ‖X_f - W_c A_f R_f‖²_F           （全局视图侧多层分解）
       + ‖A_c - A_f‖²_F                   （两条分解路径相互一致）
```

- `B` 是锚点矩阵，把每个视图重构成 `X⁽ⁱ⁾ ≈ B P⁽ⁱ⁾`（`P⁽ⁱ⁾` 为视图到锚点的映射），
  复杂度因此显著下降。
- `X_f` 是**重构出的全局视图**（注意：不是原特征拼接），由标签侧信息与锚点侧
  多层分解共同驱动，形成"锚点引导"的闭环。
- 两条分解路径 `B ≈ W_c A_c R_c` 与 `X_f ≈ W_c A_f R_f` 共享 `W_c`，
  并约束 `A_c ≈ A_f` 促使结构一致。
- 视图权重 `vᵢ` 由 `1/tr(x⁽ⁱ⁾ᵀ L_y x⁽ⁱ⁾)` 归一化，`L_y` 是**标签图**的
  拉普拉斯（knn k=20，热核）。
- 特征权重 `W` 受 `‖·‖₂,₁` 稀疏约束（`lamb`），排序取逐行 2-范数。

### 代码结构（`GRAFS.py`）

| 位置 | 内容 |
|---|---|
| `normalization(data)` | 简单 min-max 归一化 |
| `view6()` 初始化 | 锚点 `B`、`W`、`A1/A2`、`W1`、各视图 `s[i]`、`d[i]`、`p[i]` |
| 循环前 | 标签图拉普拉斯构建 + 视图权重 `nu` |
| 主循环 | 更新 `s[i]`、`p[i]` → `W` → `R1/R2` → `A1/A2` → `W1` → `B` → `X1` |
| 目标函数 | `temp1 + alpha*temp2 + beta*temp3 + gamma*temp4 + lamb*temp5` |
| 排序 | `LA.norm(W, ord=2, axis=1)`，`argsort(-w_2)` |

代码中各项：`temp1 = ‖X1 W - Y‖²_F`（标签拟合）、
`temp2 = Σᵢ νᵢ‖x[i] - B p[i]ᵀ‖²_F`（锚点重构，对应论文 `L_E` 第一项）、
`temp3 = ‖B - W1A1R1‖²_F + ‖X1 - W1A2R2‖²_F + ‖A1 - A2‖²_F`（论文式 7 三项）、
`temp4 = Σᵢ νᵢ‖X1 d[i] - x[i] s[i]‖²_F`（全局视图与视图对齐）、
`temp5 = ‖W‖₂,₁`（行稀疏）。

### 变量对照

| 论文符号 | 代码变量 | 含义 |
|---|---|---|
| `B` | `B`（列数 `kk`） | 锚点矩阵 |
| `X_f` | `X1` | 重构出的全局视图 |
| `P⁽ⁱ⁾` | `p[i]` | 视图 → 锚点映射 |
| `W_c` | `W1` | 两条分解共享的权重 |
| `A_c`、`A_f` | `A1`、`A2` | 两条路径的结构矩阵 |
| `R_c`、`R_f` | `R1`、`R2` | 对应系数矩阵 |
| `vᵢ` | `nu[i]` | 视图权重 |
| `L_y` | `Ly` | **标签图**拉普拉斯（knn k=20，热核） |
| `X⁽ⁱ⁾` | `x[i]` | 各视图特征块 |
| 选择/投影矩阵 | `s[i]`（视图侧）、`d[i]`（全局侧） | |
| `W` | `W` | 全局特征权重（排序依据） |
| `α, β, γ, λ` | `alpha, beta, gamma, lamb, kk` | 平衡参数与锚点数 |

> 说明：该模块内部 `record['method']` 原写作 `'view6'`，本次已改为 `'GRAFS'`
> 以便输出结果可读（不影响任何数值）；同理 `DHLI.py` 里原写作 `'DLHI'` 的字段
> 已修正为 `'DHLI'`。两处都只是记录用的字符串，不参与计算。

---

## 7. 关于 THBFS（只有宣传材料，无全文、无代码）

仓库里的 `pre-pdf/THBFS.pdf` 是一页**宣传材料**（现已归档在 `pre-pdf/`），
`alg/THBFS_code_description.md` 是**流程说明**，
两者都明确表示不公开关键公式与实现细节。这是六篇已发表论文之外的一份新工作，
因此本仓库**没有**它的实现代码，也无从获取全文。

- **标题**：*Scalable Multi-View Multi-Label Feature Selection via Tensor-Coupled
  Hypergraph-Bipartite Consensus*
- **作者**：Pingting Hao、Shaoqi Zhang、Huijie Zhang（东北师范大学信息科学与技术学院）
- **要解决的问题**：基于锚点的现有框架通常依赖彼此独立的成对图，把结构建模与语义
  拟合割裂，无法把跨视图高阶相关性与样本–标签依赖统一起来。
- **提出方法**：在紧凑锚点空间中联合建模结构与语义；用**低秩张量**提取跨视图共识
  （把视图特有误差与潜在结构解耦）；用**超图–二部图**结构刻画高阶关系，保持线性
  复杂度；用 **ALM + ADMM** 做交替优化；并采用掩码引导的联合学习范式，让伪标签
  同时融合精确标签监督与潜在拓扑结构。
- **材料中给出的实验**：`α = 0.05` 显著性水平下的临界值与四个指标的 Friedman 统计
  量；AP 指标在三个数据集上的消融；MIRFlickr 上的参数敏感性；VOC07 与 MIRFlickr
  上的运行时间分析。
- **代码状态**：**未提供**。若要接入本仓库的评测框架，只需新增 `alg/THBFS.py`，
  入口遵守统一签名（返回 `record['idx']` 为降序特征索引），即可通过
  `main.py` 的 `ALGORITHMS` 注册表直接调用。

---

## 8. 本文档的信息来源与核验

本文档中的算法描述来自：

- **`pre-pdf/` 下六篇论文的全文 PDF**（TOCL / UGRFS / EF2FS / DHLI / I2VSLC / GRAFS），
  以及 THBFS 的一页宣传材料；
- **6 个 `alg/*.py` 模块的实际代码**（目标函数、变量名、更新步骤均逐行核对）；
- 各论文的**官方题录**（作者、卷期、页码、DOI 取自 Crossref API，已与 PDF 内页比对）；
- Semantic Scholar 开放接口（用于补齐摘要）。

各项核验结论：

| 核验项 | 结果 |
|---|---|
| PDF 题名/作者/页数 与 Crossref 题录一致 | 6/6 通过 |
| PDF 首页页眉 DOI 与题录一致 | 4 篇 Elsevier 版全部一致 |
| 论文目标函数与代码 `objectives` 表达式逐项对应 | 6/6 通过（见各节公式与 `temp` 对照） |
| 变量对照表中的每个代码变量名 | 均在对应 `alg/*.py` 中实际存在 |

因此文中的"变量对照表"与"目标函数各项"是**论文与代码双向核对**的结果，
可直接用于阅读源码。个别处代码与论文符号命名不一致时（如 EF2FS 的
`G`↔`V`、UGRFS 的 `δ`↔`lamb`、GRAFS 的 `W_c`↔`W1`），已在该节明确标注。
