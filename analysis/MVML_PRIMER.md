# MVML 入门笔记 —— 多视图多标签学习与特征选择

> **这份笔记给谁**：刚进入 multi-view multi-label (MVML) 方向、但在别的领域做过研究的人。
> 假设你会读矩阵公式、会写 Python，但不知道这个方向在干什么、指标怎么算、论文里的符号从哪来。
>
> **怎么用**：
> 1. 先通读 §0–§2（半天），这时你应该能看懂任何一篇 MVML-FS 论文的"问题+实验"部分；
> 2. 再读 §4 的数学工具箱（一天），这时你应该能看懂目标函数和更新式；
> 3. 然后按 §7 的上手路线，在**本仓库**上跑通一遍（一天）；
> 4. 最后按 §9 的顺序读那六篇论文，配合 `analysis/PAPER_METHOD_ANALYSIS.md`。
>
> **和仓库里其它文档的关系**：
>
> | 文档 | 回答什么问题 |
> |---|---|
> | **本文（`MVML_PRIMER.md`）** | 这个领域是什么、怎么读、要会哪些数学、坑在哪 |
> | `PAPER_METHOD_ANALYSIS.md` | 六篇论文逐篇拆解 + 代码偏差 + 改进思路（**深水区**） |
> | `EXPERIMENT_PROTOCOLS.md` | 六篇的实验设置各是什么、哪里不可比 |
> | `REPRODUCE_TOCL_UGRFS.md` | 怎么把这个仓库的复现跑完、实测耗时多少 |
> | `insight/FORMULA_TO_CODE.md` | 公式 ↔ 代码逐行对照 |
> | `CODE_WALKTHROUGH.md` | **六份 `alg/*.py` 按代码执行顺序逐块解读**（含行号、公式对应、已知的坑） |
> | `README_zh.md` | 运行器（`main.py`）的参数与输出说明 |
>
> 为省地方便，下文的三个缩写：**METHOD** = `PAPER_METHOD_ANALYSIS.md`、**PROTO** = `EXPERIMENT_PROTOCOLS.md`、**REPRO** = `REPRODUCE_TOCL_UGRFS.md`。
> 不带这个前缀的「§x.y」一律指**本笔记**自己的小节。

---

## 0. 五分钟看懂这个领域

### 0.1 三个词叠起来是什么

| 词 | 一句话 | 典型例子 |
|---|---|---|
| **多标签 (multi-label)** | 一个样本**同时属于多个类**，输出是一个 0/1 向量而不是一个类别号 | 一张图同时有"沙滩/天空/人"；一篇论文同时属于"机器学习/图论" |
| **多视图 (multi-view)** | 同一个样本有**多组特征**，每组从不同角度描述它 | 一张图有 RGB 直方图 / GIST / 词袋；一篇文章有标题词袋 / 正文 TF-IDF / 引用网络 |
| **特征选择 (feature selection)** | 从 $d$ 维里挑出 $k$ 维（$k\ll d$），**只输出排序，不输出分类器** | 从 1312 维里挑 20% = 262 维，再交给下游分类器 |

叠起来就是本文档的主角：**MVML-FS = 在多视图多标签数据上给特征排序**。

### 0.2 形式化（这一节请务必看懂，后面全是它的变体）

给定

* 视图：$X^{(i)}\in\mathbb{R}^{n\times d^{(i)}}$，$i=1,\dots,V$（把它们按列拼起来记作 $X\in\mathbb{R}^{n\times d}$，$d=\sum_i d^{(i)}$）
* 标签：$Y\in\{0,1\}^{n\times l}$

要输出：一个长度 $d$ 的**特征排序** $\pi$（外加"每个特征来自哪个视图"的附带信息）。

所有方法都做同一件事：**学一个权重矩阵 $W\in\mathbb{R}^{d\times l}$，然后按行范数 $\|W_{(j)}\|_2$ 从大到小排序**（§4.2 会解释为什么是行范数）。

### 0.2.1 六篇共用的目标函数模板（这一支的"万能句式"）

把六篇的目标函数摆在一起看，会发现它们都在套同一个模板：

$$
\min_{\text{待学变量}}\ \;
\underbrace{\mathcal L_{\text{fit}}}_{\text{主拟合项：系数固定为 }1}
\;+\;\alpha\,\mathcal C_1\;+\;\beta\,\mathcal C_2\;+\;\gamma\,\mathcal C_3
\;+\;\delta\underbrace{\lVert W\rVert_{2,1}}_{\text{稀疏 → 排序}}
\tag{0.1}
$$

读这个模板要抓住四件事：

1. **主拟合项的系数固定为 1**。不是"论文懒得调"，而是不固定就**不可辨识**——把整个模型乘 2 不改变最优解的结构，主项的尺度必须钉死，其余系数才有意义。
2. **因此自由系数恰好 4 个**：$\alpha,\beta,\gamma$ 是三个"结构约束"的权重，$\delta$ 是稀疏项的权重。这也解释了 §5.3 里"7 个取值 × 4 个参数"的由来——**4 是模板的产物，不是领域规定**（换一篇论文可能只有 2 个，也可能有 6 个）。
3. **$\mathcal C_1,\mathcal C_2,\mathcal C_3$ 里放什么，就是这篇文章的全部创新**。这一支的论文不会去改主拟合项的形状，而是往这三个槽位里塞新的结构约束。
4. **$\delta$ 在六篇里的五篇都是行稀疏 $\lVert W\rVert_{2,1}$**（唯一例外是 DHLI，它的 $\delta$ 管视图互斥）。这是模板里最稳定的一个槽位。

三个槽位里的内容可以归成下面几类（**① 就是主拟合项本身，不占槽位**）：

| 类别 | 典型形式 | 在解决什么 |
|---|---|---|
| **② 图/流形平滑** | $\mathrm{Tr}(F^\top LF)$ | 让"相似样本的表示也相似"（§4.3） |
| **③ 一致性/重构** | 全局视图 ⇄ 各视图、双路映射互相靠近 | 让多个视图既共享又互补 |
| **④ 标签结构** | 标签相关矩阵、hybrid labels、显式噪声项 | 把标签之间/标签内部的结构当额外监督 |
| **⑤ 稀疏** | $\ell_{2,1}$ 行稀疏（固定占 $\delta$ 槽） | 从权重矩阵产出**排序** |

> **拿到任何一篇新论文，用这个模板三步拆**：
> **①** 先找系数为 1 的那一项（主拟合项）⇒ 定下"谁拟合谁"（METHOD §2 的"轴 III：损失拓扑"讲的就是这条轴）；
> **②** 把带 $\alpha/\beta/\gamma$ 的三项各自命名 ⇒ 这三项就是论文的创新点；
> **③** 找 $\delta\lVert W\rVert_{2,1}$ ⇒ 确认排序量确实是 $W$ 的行范数。
> 三步走完，这篇论文的骨架就出来了。逐篇展开见 §5.3 的参数对照表与 METHOD §3；
> **模板的正式表述（式(1.2) 与"四个字母各管哪一项"的深水版）见 METHOD §1.2。**

### 0.3 为什么难：三个内在矛盾

1. **共性 vs 互补**：各视图描述同一批样本 ⇒ 既有冗余（该扔）又有各自独有的信息（该留）。怎么把两者分开，是这个方向一半论文的主题。
2. **标签噪声 vs 监督信息不足**：多标签数据多来自众包/自动标注，$\{0,1\}$ 里既有假阳也有假阴；而标签本身又是唯一的监督信号，去噪去过头就没东西可学了。
3. **排序目标 vs 连续优化**：真正的目标是"选一个子集"（离散、NP-hard），所有方法都把它**松弛**成连续稀疏优化，于是"稀疏程度"和"最终排序质量"并不天然一致。

> **判断一篇论文属不属于这一支**：看它的目标函数有没有
> ①一个"特征→标签"的拟合项（$\|XW-Y\|$ 型）、②一个行稀疏项（$\ell_{2,1}$）。
> 两条都有 ⇒ 嵌入式稀疏回归（本方向最大的一支，仓库里六篇全在内）；
> 若写的是互信息、分类间隔、或纯重构误差 ⇒ 属于别支（见 §3）。

### 0.4 这个领域的"最小可信论文"长什么样

一篇 MVML-FS 论文的实验部分基本固定是这一套（六篇都一样）：

```
6 个数据集 × 5 折交叉验证 × 特征比例 1%..20%
   ↓ 每个 (数据集, 折, k) 组合：
特征选择算法输出排序 → 取前 k 个特征 → 训一个标准多标签分类器（MLkNN）→ 算 4 个指标
   ↓
报"折平均 ± 标准差"，再补 Friedman 检验 / 参数敏感性 / 消融 / 收敛曲线
```

**看清楚这条链**：特征选择算法本身**不预测任何东西**，它只给排序；预测和指标全部由下游 MLkNN 完成。
这就是为什么"特征选择论文的 SOTA 差距经常落在标准差内"——排序只要大体对，MLkNN 就能把差距吃掉。

---

## 1. 任务、记号与数据

### 1.1 记号表（与六篇论文对齐）

| 符号 | 含义 | 维度 |
|---|---|---|
| $n$ | 样本数 | 标量 |
| $V$ | 视图数 | 标量 |
| $d^{(i)}$ | 第 $i$ 个视图的特征数；$d=\sum_i d^{(i)}$ | 标量 |
| $l$（有些论文写 $c$） | 标签数 | 标量 |
| $X^{(i)}$ / $X$ | 第 $i$ 个视图 / 拼接后的全部特征 | $n\times d^{(i)}$ / $n\times d$ |
| $Y$ | 观测多标签矩阵，$Y_{pj}=1$ 表示样本 $p$ 有标签 $j$ | $\{0,1\}^{n\times l}$ |
| $W$ | **要学的**特征权重（排序依据） | $d\times l$ |
| $y^{(i)}$ / $y_t^{(i)}$ | 视图特有（伪）标签 | $n\times l$ |
| $L^{(i)}$ | 图拉普拉斯矩阵 | $n\times n$ |
| $\|W\|_{2,1}$ | 行组稀疏范数 $\sum_p\sqrt{\sum_q W_{pq}^2}$ | 标量 |
| $\alpha,\beta,\gamma,\delta$ | **权衡系数**（模板 §0.2.1 的槽位权重）；在代码里 $\delta$ 叫 `lamb` | 标量（**逐篇含义不同**，见 §5.3） |

> ⚠️ **本方向的记号不统一**，这是初学最大的障碍之一。至少要知道：
> * $c$ 和 $l$ 都常用来表示标签数；
> * $U,V$ 在不同论文里可能是"隐标签矩阵 / 投影矩阵"，也可能是**视图数**，还可能（EF²FS 代码里）是**隐标签 $G$**；
> * $D$ 在 UGRFS 里是"全局视图分布"，在 GRAFS 里是"视图切分算子"，在别的论文里是"$\ell_{2,1}$ 的重加权对角阵"。
> **读每篇前先看它的 Notation 表，不要带上一章的记号惯性。**

### 1.2 本仓库的 10 个数据集（实测维度，非抄论文）

下面这张表是用仓库自己的加载器 `main.get_data()` 跑出来的（`data/*.mat`）：`card` = 平均标签基数（每个样本平均带几个标签），`dens` = card / l（标签密度），`rare` = 正例比例 < 5% 的标签个数。

| 仓库键 | n | V | d | l | 视图维度 | card | dens | rare |
|---|---|---|---|---|---|---|---|---|
| `emotions` | 593 | 2 | 72 | 6 | [8, 64] | 1.87 | 0.311 | 0 |
| `yeast` | 2417 | 2 | 103 | 14 | [79, 24] | 4.24 | 0.303 | 1 |
| `3sources` | 169 | 3 | 3000 | 6 | [1000, 1000, 1000] | 1.14 | 0.190 | 0 |
| `SCENE` | 4400 | 5 | 634 | 33 | [64, 225, 144, 73, 128] | 6.49 | 0.197 | 13 |
| `OBJECT` | 6047 | 5 | 634 | 31 | [64, 225, 144, 73, 128] | 2.03 | 0.066 | 18 |
| `VOC07` | 3817 | 3 | 712 | 20 | [100, 512, 100] | 2.21 | 0.111 | 7 |
| `MIRFlickr` | 4053 | 3 | 712 | 38 | [100, 512, 100] | 9.11 | 0.240 | 13 |
| `corel5k_5` | 4999 | 5 | 1312 | 260 | [100, 300, 512, 100, 300] | 3.40 | 0.013 | 249 |
| `iaprtc12` | 4999 | 5 | 1312 | 291 | [100, 300, 512, 100, 300] | 5.04 | 0.017 | 269 |
| `espgame` | 4999 | 5 | 1312 | 268 | [100, 300, 512, 100, 300] | 4.91 | 0.018 | 252 |

**从这张表能读出这个领域的三个现实**：

1. **规模跨度极大**：$n$ 从 169 到 6047，$l$ 从 6 到 291，$d$ 从 72 到 3000。所以论文里"我们的方法在所有数据集上都最好"这句话的检验强度，取决于数据集选择。
2. **标签密度极端稀疏**：`iaprtc12` 只有 1.7% 的位置是 1，而 `emotions` 有 31%。密度越低，**随机猜"全 0"的 Hamming Loss 就越低**，指标越容易被"偷分"——所以必须同时看 AP / Ranking Loss 这类排序指标（§2）。
3. **长尾标签普遍**：`corel5k_5` 有 249/260 个标签正例不足 5%。这类标签几乎是不可学的，但它们会**大幅主导 Coverage / Ranking Loss**。

### 1.3 怎么自己读一个 `.mat`

```python
import scipy.io as io, numpy as np
mat = io.loadmat("data/MIRFlickr.mat")
print([k for k in mat if not k.startswith("__")])   # ['view', 'label', 'features', ...]
view = mat["view"][0]          # 长度 V 的数组，view[i] 是 n × d_i
print(view[0].shape, view[1].shape, view[2].shape)  # (4053, 100) (4053, 512) (4053, 100)
print(mat["features"], mat["label"].shape)          # [[100 512 100]]  (4053, 38)
```

仓库统一约定：`view` 是"每视图一块"、`features` 是各块宽度、`label` 是标签矩阵，`main.get_data()` 会把它们拼成 `X, view_dims, Y` 并校验维度自洽。

> ⚠️ **一个真实的坑**：5 视图数据集（`iaprtc12` / `corel5k_5` / `espgame`）的 `.mat` 里，第 4、5 个视图的顺序与论文 Table 1 写的是**反的**（`.mat` 是 `[100,300,512,100,300]`，论文写 `DH,DHV3H1,GIST,HHV3H1,HH = [100,300,512,300,100]`）。
> 维度总和一样所以不报错，但 TOCL 这类**沿某一维做 FFT** 的方法对切片顺序敏感（实测排名相关只有 0.456）。
> 详见 `REPRODUCE_TOCL_UGRFS.md` §1.4，运行时可加 `--view-order 0,1,2,4,3` 对齐论文。

---

## 2. 评价指标：这个领域的"语言"

### 2.1 两类指标

| 类别 | 需要什么 | 例子 | 评价的是 |
|---|---|---|---|
| **label-based**（基于标签决策） | 分类器给的 0/1 预测 | Hamming Loss、0/1 Loss、(micro/macro-)F1 | "最终判对没有" |
| **ranking-based**（基于标签排序） | 分类器给的**每个标签的分数** | Average Precision、Coverage、Ranking Loss | "排序好不好" |

多标签论文**两类的混合**是很常见的：TOCL/UGRFS 报的四个指标里，AP/Coverage/RL 是 ranking-based，HL 是 label-based。

### 2.2 四个论文指标（定义、直觉、以及最容易搞错的口径）

设测试样本 $p$ 的真实标签集 $S_p\subseteq\{1..l\}$，分类器给出每个标签的分数 $f_p(j)$（越大越可能为正）。把标签按分数**降序**排成一个序列。

| 指标 | 定义（单个样本） | 直觉 | 方向 |
|---|---|---|---|
| **AP** (Average Precision) | $\frac{1}{\lvert S_p\rvert}\sum_{j\in S_p}\frac{\#\{j'\in S_p:\ \text{rank}(j')\le \text{rank}(j)\}}{\text{rank}(j)}$ | 正标签是否被排在前面（类似信息检索的 MAP） | ↑ 越大越好 |
| **Coverage** | 从序列头部往下扫，**扫到覆盖住 $S_p$ 全部正标签**所需的标签个数 | 要翻多少个标签才能不漏 | ↓ 越小越好 |
| **Ranking Loss** | 反序对的平均比例 $\frac{\lvert\{(j,j'): j\in S_p, j'\notin S_p, f(j)<f(j')\}\rvert}{\lvert S_p\rvert\cdot(l-\lvert S_p\rvert)}$ | 正标签被负标签压在后面的比例 | ↓ 越小越好 |
| **Hamming Loss** | $\frac{1}{l}\lvert\{j: \hat Y_{pj}\ne Y_{pj}\}\rvert$ | 逐标签判错的平均比例 | ↓ 越小越好 |

**三个必须知道的坑**：

1. **Coverage 的归一化口径**。`sklearn.metrics.coverage_error` 返回的是"个数"，但论文表里的 Coverage 是**除以标签数 $L$** 的值。仓库为此专门做了换算（`main.py` 里的 `CV_paper_mean_norm = CV_paper_mean / n_labels`），并用 all-features MLkNN 反推验证过：只有除以 $L$ 才能落在论文的量级（SCENE 0.411 vs 论文 0.419）。
   **不问清口径就复现，会得到"差 30 倍"的假结论。**
2. **AP 的聚合口径**。论文在 1%–20% 的特征比例上各算一次，然后**报一个数**——但"怎么从 20 个数变成一个数"（平均？取最好？取 20% 那个点？）论文**没有写**。仓库的做法是取 20 个点的平均（`main.paper_aggregate`），并注明"这是推断"。
3. **报"最好值"还是"某个 k 的值"是两件事**。仓储里同时输出 `AP_best`（曲线最大值）、`AP_auc`（曲线下面积）、`AP_paper_mean`（论文口径）。**对比论文只能用第三个**，前两个是仓库自己加的诊断量。

### 2.3 为什么要扫 1%–20% 而不是报一个点

因为"选多少特征"本身不是模型决定的：曲线告诉你方法在小 $k$ 区间的**早期发现能力**（这恰恰是特征选择的价值所在）。只报一个 $k$ 的结果，既容易被 $k$ 调参偷分，也掩盖了方法之间的差异形态。

### 2.4 显著性检验

六篇里 **TOCL**（Table 4：AP 6.629 / Coverage 9.100 / HL 9.730 / RL 6.248，临界值 2.244）与 **EF²FS**（Table 3–5：AP 4.368 / Coverage 3.435 / HL 4.912 / RL 4.000，临界值 2.53）报了 Friedman 统计量；UGRFS 则用三变体消融代替。Friedman + Nemenyi 是多数据集多算法比较的标准做法（Demšar 2006）：

1. 每个数据集上给所有方法排名（1 = 最好）；
2. 对每个方法求平均排名，做 Friedman 检验判断"是否至少有一个方法显著不同"；
3. 若显著，用 Nemenyi 算临界差 CD，画 CD 图看哪些方法对之间差异显著。

**读论文时注意**：Friedman 显著，只说明"这批方法的平均排名不完全相同"，**不等于**"你和冠军有显著差异"——判断两两差异要看 Nemenyi 的临界差（CD）。六篇表格里"比基线高一点点"的情况，多数落在标准差内（逐篇的幅度核算见 METHOD §3 各篇的「实验证据」小结）。

### 2.5 一个自检

> 如果某方法把 `iaprtc12` 的 Hamming Loss 从 0.0155 降到 0.0153，算提升吗？
>
> 先算"什么都不学"的基线：**全 0 预测的 HL 恰好等于标签密度**，即 0.0173。
> 于是 ① 这个数字本身有信息（比全 0 好约 10%）；② 但 0.0002 的差距只有相对 1.3%，
> 而且 HL 是**逐元素**平均、被 291 个标签里那 269 个稀有标签（正例 < 5%）主导。
> 结论：单看 HL 判断不了方法好坏，必须同时看 AP / Ranking Loss 这类排序指标。
>
> **顺手记住这个技巧：任何表格里的 HL 都能立刻估出"水分"——HL 的全 0 基线就是该数据集的标签密度**（§1.2 的表已给出）。

---

## 3. 方法地图：这个领域有哪些流派

| 流派 | 目标函数长相 | 代表工作 | 优点 / 缺点 | 本仓库 |
|---|---|---|---|---|
| **① Filter：信息论/相关性** | 不学 $W$；算互信息、最大相关最小冗余（mRMR/MIFS 一脉） | 多标签版的 mRMR/PMU 等 | 快、与分类器无关；但忽略特征组合、不能联合优化 | ✗ |
| **② Filter：谱/流形打分** | 用 $X$ 与 $Y$ 的图或散度给每个特征打标量分（Laplacian Score、Fisher Score） | lsPCA、SPEC 等 | 极快；无监督或弱监督，预测力有限 | ✗ |
| **③ Embedded：稀疏回归**（**主流**） | $\min_W\ \|XW-Y\|_F^2+\lambda\Theta(W)+\text{(结构约束)}$，$\Theta$ 含 $\ell_{2,1}$ | RFS、MDFS、MVML 稀疏 FS、**本仓库六篇** | 排序质量好、可联合建模；非凸/难解、需调参 | ✅ 六篇 |
| **④ Embedded：自表达/子空间** | $X\approx XZ$，对 $Z$ 加行稀疏（常无监督） | UFS 系列、自表达特征选择 | 不需要标签；但目标与下游任务间接 | ✗ |
| **⑤ Embedded：NMF/概念分解** | $X\approx UV$，用 $U$ 或 $V$ 的行选择 | 各类 NMF-FS | 可解释（基=概念）；需非负假设 | ✗（但六篇的**乘性更新**来自这里） |
| **⑥ Wrapper / 进化算法** | 直接搜索子集，目标函数是下游分类器性能 | 遗传算法、粒子群 + 多标签 | 直接优化目标；计算代价极高、易过拟合验证集 | ✗ |
| **⑦ 深度学习** | 端到端表示 + 注意力/对比学习 | 近年的深度 MVML（视图缺失、部分标签） | 表达能力强；需要大数据、可解释性弱 | ✗ |

> **怎么定位自己**：如果你的想法是"换个更好的稀疏正则 / 加一个新的结构约束"，落点在 ③；
> 如果是"用互信息挑特征"，落点在 ①。仓库六篇全部在 ③，且都在"结构约束"上做增量（见 `PAPER_METHOD_ANALYSIS.md` §1.2、§2）。

---

## 4. 读懂这六篇所需的最小数学工具箱

这一节是**你真正需要补的课**。每一条都给"是什么 / 诱导什么结构 / 在本仓库哪里出现"。

### 4.1 范数一族

| 范数 | 定义 | 诱导的结构 | 近端算子（prox） | 出现处 |
|---|---|---|---|---|
| $\ell_1$ | $\sum_{pq}\lvert W_{pq}\rvert$ | 逐元素稀疏 | 软阈值 $\max(w-\tau,0)$ | DHLI 的 $\|Y_n\|_1$ |
| $\ell_2$（Frobenius） | $\sqrt{\sum_{pq}W_{pq}^2}$ | 整体缩小，**不稀疏** | $w/(1+\tau)$ | 各篇的正则 $\|U\|_F^2$ |
| **$\ell_{2,1}$** | $\sum_p\sqrt{\sum_q W_{pq}^2}$ | **整行一起为 0** ⇒ 特征级选择 | 行软阈值 $\max(\|W_p\|_2-\tau,0)\cdot\frac{W_p}{\|W_p\|_2}$ | 六篇的排序项 |
| $\ell_{2,p}$ / Schatten-$p$ | 奇异值的 $\ell_p$ | $p<1$ 更稀疏、$p=1$ 核范数 | — | Schatten-$p$ 变体 |
| **核范数** $\|\cdot\|_*$ | 奇异值之和 | 矩阵低秩 | 奇异值软阈值（SVD） | TOCL 的张量版（t-SVD） |

**一句话记忆**：$\ell_1$ 稀疏"元素"、$\ell_{2,1}$ 稀疏"行"、核范数稀疏"奇异值/秩"。
特征选择要的是**行**稀疏，因为"一个特征"就是一整行。

### 4.2 为什么排序量是 $\|W_{(j)}\|_2$（而不是 $|W_{pq}|$）

多标签里 $W\in\mathbb{R}^{d\times l}$：一行 = 一个特征，一列 = 一个标签。三条理由：

1. **组结构**：$\ell_{2,1}$ 把一行当作"一个组"，近端解会把整行压成 0 —— 这才对应"这个特征选不选"；逐元素 $\ell_1$ 会保留"只对部分标签有用"的特征。
2. **尺度稳健**：不同标签（列）的权重尺度差别很大，逐元素比较没有意义；行 2-范数把一行内各标签的分量合成一个可比的标量。
3. **和 $\ell_{2,1}$ 一致**：$\|W\|_{2,1}=\sum_p\|W_p\|_2$ 本身就是"行范数之和"，排序量就是它的加数。

> 变体：有些实现会**先对 $W$ 的列归一化**再算行范数；本仓库 DHLI 是先对 $W$、$U$ 各自 MinMax 再相加（`alg/DHLI.py` L202–212）。这些差异会让同一个模型给出不同排序。

### 4.3 图拉普拉斯与流形正则（六篇共用）

**是什么**：给定 $n$ 个样本的 knn 图（相似度矩阵 $S$，$s_{pq}\ge0$），度数矩阵 $A=\mathrm{diag}(\sum_q s_{pq})$，则 **$L=A-S$**（$A-S$ 是组合拉普拉斯；归一化的 $I-A^{-1/2}SA^{-1/2}$ 也常见）。

**为什么写作 $\mathrm{Tr}(Y^\top LY)$**：由平滑假设

$$
\frac12\sum_{p,q}s_{pq}\big\|y_p-y_q\big\|^2=\mathrm{Tr}(Y^\top LY)
$$

（"图上相邻的样本，其表示应当相近"）。这条恒等式是流形正则的全部内容，值得自己推一遍——左式展开后用 $A,S$ 的定义合并即得。

**在本方向的用法**：把 $Y$ 换成"要平滑的量"，同时**必须问清图建在哪**。仓库六篇的实际组合是：

| 论文 | 图建在 | 正则作用在 | 代码里的变量 |
|---|---|---|---|
| I²VSLC | 视图特征 | 视图特有标签 $y^{(i)}$ | `Lx_lst[i]` |
| UGRFS | **标签** | 全局视图 $D$ | `Ly`（`construct_W(Y)`） |
| EF²FS | 视图特征 | 隐标签 $G$ | `Lx_lst[i]` |
| GRAFS | 论文写视图特征，**代码建的是标签图** | 视图权重 $v_i$ | `Ly`（METHOD §5.2 GRAFS 行） |

同一句"图正则"，建图空间不同，语义完全不同 —— 这是读这类论文最容易混过去的一处。

**三个实践细节（也是坑）**：

* 图怎么建：常用 `knn + 热核`（$s_{pq}=\exp(-\|x_p-x_q\|^2/2t^2)$）。**$t$ 是尺度参数，必须与数据尺度匹配**——六篇把它写死成 $t=1.0$，在真实数据上会热核下溢（§6.2）。
* 六篇都用 `skfeature.utility.construct_W`，这是 scikit-feature 的工具函数。
* 图建在**特征空间**还是**标签空间**，语义完全不同；仓库里 UGRFS 与 I²VSLC 恰好一个建在标签上一个建在特征上，可以对照着看。

### 4.4 乘性更新：非负假设下的"一步 KKT"

六篇的优化器都是同一套：把除当前变量外的都固定，然后

$$
Z\ \leftarrow\ Z\circ\frac{[\nabla\Theta]^-}{[\nabla\Theta]^+}
$$

（$\circ$ 是 Hadamard 积）。它不是启发式，推导只有三行：

1. 所有数据和变量非负（六篇的数据是直方图/GIST/词袋这类非负特征；
   **若数据含负值，分子可能为负、乘性更新会破坏非负假设甚至发散**，METHOD §5.2 的 GRAFS 行有记录）；
2. 把子问题梯度按符号拆开：$\nabla f(Z)=\nabla^+f-\nabla^-f$，两项逐元素非负；
3. 对 $\min f(Z)\ \mathrm{s.t.}\ Z\ge0$ 写 KKT：$Z\circ\nabla f(Z)=0$，即 $Z\circ(\nabla^+-\nabla^-)=0$，
   于是 $Z=Z\circ\nabla^-/\,\nabla^+$。

**读更新式的规则**：分子里的加项 = 让该变量增大的力；分母里的加项 = 让它减小的力。

**三个必须记住的后果**：

* 没有步长（不用选学习率），代价是步长由数据尺度决定；
* **$0$ 是吸收态**：被更新成 0 的元素永远回不来（所以代码里到处是 `eps`）；
* **没有收敛保证**：乘性更新只在能写出辅助函数的问题上单调下降，六篇**都没有定理**（`PAPER_METHOD_ANALYSIS.md` §5.7、§6.7）。

### 4.5 其它四个必备概念（够读论文即可）

| 概念 | 一句话 | 在本仓库 |
|---|---|---|
| **伪标签 / view-specific label** | 把"每个视图应该有自己的标签"变成可学变量，再用"它们的并集 = 观测标签"约束 | I²VSLC 首创（$y^{(i)}$）、DHLI 升级为 hybrid labels、TOCL 加"两层"定义 |
| **一致性 + 互补性** | 视图之间既要一致，又必须保留各自的独有信息；做法有：共享隐空间、显式分解、对立互补双路映射 | 从 MVLD 到 UGRFS/TOCL 的整条演进线（METHOD §2 的"轴 II：视图拓扑"） |
| **低秩 / 核范数 / t-SVD** | 用"奇异值之和"逼近"秩"，是去噪与结构共享的通用工具；张量版需要先沿 tube 方向做 FFT | TOCL 的加权张量核范数（`PAPER_METHOD_ANALYSIS.md` §3.1） |
| **ADMM / 交替方向法** | 把"难解的整体问题"拆成几个有闭式解的子问题 + 乘子更新；相比乘性更新**有收敛理论**（凸情形） | 仓库最前沿的 THBFS 明确改用 ALM+ADMM；§6.7 给了可验证的收敛判据 |

### 4.6 一个"我读懂了没"的判据

给你一篇新论文的目标函数，你应该能在 10 分钟内答出：

1. 除了 $W$，它还引入了哪些**可学变量**？各自维度？
2. 每一项在**约束谁**（拟合 / 稀疏 / 结构）？
3. 排序最终用哪个量？
4. 它的优化器有收敛保证吗？

答不出第 1 条，说明记号没理顺；答不出第 4 条，说明你还没进入审稿人视角。

---

## 5. 实验协议：怎么做一份别人挑不出毛病的实验

### 5.1 划分

* **5 折交叉验证**是这个方向的默认（六篇都是）；"每折"指把数据分成 5 份，轮流留 1 份做测试。
* **折数必须在所有方法间一致**，且折的划分要固定（同一份数据、同一个随机种子）。
* 注意"**shuffle 之前 X 和 Y 必须一起动**"——仓库早期版本 shuffle 了 X 却没 shuffle Y，直接让特征与标签错位（`main.py` L184–214 的注释就是记录这个 bug）。

### 5.2 特征比例扫描

* 论文口径：1%–20%，步长 1%（`PAPER_PCTS`）。
* 每个方法都要在**同一个** $k$ 上比较；曲线图比单点更有说服力。
* 报"最好值"时必须同时报对应的 $k$，否则无法判断是不是调出来的。

### 5.3 参数网格：那 4 个参数是什么

它们是**权衡系数（trade-off / balance parameters）$\alpha,\beta,\gamma,\delta$**——即 §0.2.1 模板里除主拟合项之外的那几个槽位权重。论文原文：

> **DHLI**（AAAI'24）："In DHLI method, there are **four trade-off parameters α, β, γ and δ** that influence the performance results. The parameter is **individually tuned while keeping the other parameters fixed**, and the grid search is conducted over a predefined range."
>
> **TOCL**（MM'25）："The TOCL framework incorporates **four parameters α, β, γ and δ** into two key components… We **tune each parameter individually**."

> ⚠️ **这 4 个字母在六篇里不是同一件事** —— 同一个 $\alpha$ 在不同论文里管的是完全不同的项。
> **读任何一篇之前先查下面这张表**（它是 METHOD §3 各篇"逐项解剖"的浓缩版）：

| 论文 | $\alpha$ 乘的项 | $\beta$ 乘的项 | $\gamma$ 乘的项 | $\delta$ 乘的项 |
|---|---|---|---|---|
| **TOCL** | 全局非线性映射 $\lVert PXW-FU\rVert^2+\lVert U\rVert^2+\lVert V\rVert^2$ | $\lVert P\rVert_F^2$（$P$ 的尺度） | 离散标签相关 $\mathcal P_2$ | $\lVert W\rVert_{2,1}$ 行稀疏 |
| **UGRFS** | 标签图平滑 $\mathrm{Tr}(D^\top L_YD)$ | 全局视图拆回各视图 $\sum_i\lVert D^{(i)}-\mathrm{diag}(C^{(i)})X^{(i)}\rVert^2$ | 与加权拼接一致 $\lVert D-X^f\rVert^2$ | $\lVert W\rVert_{2,1}$ 行稀疏 |
| **EF²FS** | 隐标签回归标签 $\lVert Y-GB^\top\rVert^2$ | 各视图拟合公共嵌入 $\sum_i\lVert x^{(i)}w^{(i)}-G\rVert^2$ | 全局块↔局部权重一致 $\sum_i\lVert a^{(i)}-w^{(i)}\rVert^2$ | $\lVert A\rVert_{2,1}$ 行稀疏 |
| **DHLI** | 公共标签占多数 $\lVert Y\ominus Y_c\rVert^2$ | 噪声稀疏 $\lVert Y_n\rVert_1$ | $\lVert W\rVert_{2,1}+\lVert U\rVert_{2,1}$ | **视图互斥 + 并集一致**（⚠️ 不是稀疏） |
| **GRAFS** | 锚点重构各视图 $\sum_i v_i\lVert X^{(i)}-BP^{(i)}\rVert^2$ | 双路径分解 + $A^c\!\approx\!A^f$ 软一致（$L_E$） | 全局视图逐视图还原 $\sum_i v_i\lVert X^fD^{(i)}-X^{(i)}C^{(i)}\rVert^2$ | $\lVert W\rVert_{2,1}$ 行稀疏 |
| **I²VSLC** | 双层标签关系 $\Phi(\cdot)$（视图内流形 + 视图间共识） | 标签相关增强 $\lVert YC-Y_{vs}\rVert^2+\lVert Y-YC\rVert^2$ | **视图级**组收缩 $\sum_i\lVert w^{(i)}\rVert_F$ | **特征级**行稀疏 $\lVert W\rVert_{2,1}$ |

**两条能记住的规律**：

* **$\delta$ 在 6 篇里有 5 篇是"稀疏项"**（TOCL / UGRFS / EF²FS / GRAFS / I²VSLC），**唯一例外是 DHLI**——它的 $\delta$ 管视图互斥，稀疏放在 $\gamma$ 里。所以"δ 就是 $\ell_{2,1}$ 权重"这个口诀**只对五篇成立**。
* **最不稳定的是 $\gamma$**：在 DHLI/I²VSLC 里是稀疏、TOCL 里是标签相关、EF²FS/UGRFS 里是一致性、GRAFS 里是逐视图还原。
* 结论：**不要记字母，记"这一项在管什么"**。

**搜索方式与复现现状**：

* 范围是 $\{10^{-3},10^{-2},\dots,10^{3}\}$ ⇒ **7 个取值**。但**论文里做的是"逐个参数扫描"**（固定其余、扫一个、取最好、再扫下一个），而不是 $7^4=2401$ 的四维全网格——后者的代价远超 5 折评估本身。仓库 `tune.py` 复刻的正是扫描（`--rounds` 重复整轮就等价于坐标下降）。
* **论文没有逐数据集公布"最终选中的那组值"**，只有搜索范围 + 敏感性曲线（Fig. 3/4）。因此仓库 `DEFAULT_PARAMS` 六篇**全是 `1.0`**，这是**未经调参的占位值**；`EXPERIMENT_PROTOCOLS.md` 明确警告"必须调参才可比"。量级参考：TOCL 在 `3sources` 上用全 1.0 得 AP **0.4463**，论文是 **0.4883**。
* 报"最好值"时必须交代口径：按**折平均聚合值**选参数，还是按**测试集最好值**选（后者是测试集泄漏）。`tune.py` 默认用论文口径（1%–20% 上取平均聚合），可用 `--aggregate` 改。

**还有一类参数"不在网格里"**（论文没报、代码写死 —— 这是复现空洞所在）：

| 论文 | 写死的系数 / 结构参数 | 含义 | 位置 |
|---|---|---|---|
| TOCL | `delta=1`、`aaa=1`、`rho=1` | 局部线性项系数、张量项系数、加权 TNN 的 $C$ | `alg/TOCL.py` L77–79 |
| GRAFS | `k1=10`（硬编码）、`kk=20`（默认） | 潜锚点维数、锚点数 —— **论文两个值都没报** | `alg/GRAFS.py` L46、L22 |
| EF²FS | `V_dim=30`（须由调用者传入） | 隐空间维数 $k$ —— **论文全文未给取值** | `main.py` `DEFAULT_PARAMS` |

**命名陷阱（读代码必看）**：

| 论文符号 | 六篇代码里的统一名字 |
|---|---|
| $\alpha,\beta,\gamma$ | `alpha`, `beta`, `gamma` |
| $\delta$ | **`lamb`**（是 $\delta$，不是 $\lambda$） |
| TOCL 另有写死项 | `delta` = **局部项系数**（⚠️ 不是论文的 $\delta$）、`aaa` = 张量项系数、`rho` = 加权 TNN 的 $C$ |

### 5.4 报什么

标准套餐：**主表（6 数据集 × 4 指标，mean±std）+ 参数敏感性 + 消融 + 收敛曲线 + 复杂度/耗时**。
其中**消融最容易露怯**：六篇里有几篇的消融只做"删掉某项"，而这无法证明"在损失里融合优于先融合再选"（`PAPER_METHOD_ANALYSIS.md` §3.3.3）。

### 5.5 本仓库怎么跑（Windows / 本机 `.venv`）

```powershell
# 0) 看有哪些数据集和算法
.venv\Scripts\python.exe main.py --list

# 1) 全量冒烟：10 个数据集 × 6 个算法，各跑一个 200 样本切片。
#    ⚠️ 这**不是**"几分钟"的事（共 60 个组合，本机跑 20 分钟仍未结束）。
#    只想先确认流程通，缩小规模：
.venv\Scripts\python.exe main.py --check --check-samples 60

# 2) 最快的一个真实实验：DHLI / emotions / 1 折 / 5% 特征，本机实测 ≈ 1.2 秒
.venv\Scripts\python.exe main.py --alg DHLI --data emotions --folds 1 --select-ratio 0.05

# 2b) 完整一点：emotions / 5 折 / 20% 特征（仍然是小数据集上最便宜的组合之一）
.venv\Scripts\python.exe main.py --alg DHLI --data emotions --folds 5 --select-ratio 0.2

# 3) TOCL 很慢，先加迭代上限（见下面的警告）
$env:MVML_MAX_ITER=60; .venv\Scripts\python.exe main.py --alg TOCL --data 3sources --folds 5 --select-ratio 0.2

# 4) 与论文表格对齐地查看
.venv\Scripts\python.exe paper_table.py --alg TOCL UGRFS
```

输出文件（`results/`）：

| 文件 | 内容 |
|---|---|
| `rankings_<alg>_<dataset>.csv` | 完整特征排序 + **每个特征来自哪个视图** |
| `metrics_<alg>_<dataset>.csv` | "指标 vs 选了多少特征"的曲线（600 行左右是正常的） |
| `summary_<alg>.csv` / `summary_all.csv` | 每个数据集一行：`*_best` / `*_auc` / `*_paper_mean` 三套口径 |

> ⚠️ **`MVML_MAX_ITER` 是限流，会改变结果**：TOCL 的停机判据是相对变化 < $10^{-6}$，在 `3sources` 上 6 轮就收敛，在 `MIRFlickr` 上跑满 60 轮也不停。加限流是为了能在笔记本上跑完，但数值会与论文有差。
>
> ⚠️ **耗时预期**（实测，见 `REPRODUCE_TOCL_UGRFS.md` §6）：TOCL 在 MIRFlickr 上 **18 分钟/折**、SCENE **27 分钟/折**；OBJECT 5 折约 40 小时。**优先挑 `emotions` / `3sources` / `yeast` 练手。**

---

## 6. 七个必踩的坑（都是本仓库实测出来的）

| # | 坑 | 现象 | 为什么 | 详见 |
|---|---|---|---|---|
| 1 | **目标值曲线不能当收敛证据** | 曲线看着"快速下降后稳定"，实际在震荡 | 六篇统一画 $(z^{t-1}-z^t)/z^{t-1}$，而**震荡序列的相邻差比单调序列更大**，越震荡曲线越好看。实测 EF²FS 目标值末轮是最优值的 38.5 倍 | METHOD §5.7 |
| 2 | **热核下溢** | 某视图凭空独吞全部权重 | 图带宽写死 `t=1.0`，当中位平方距离是 705 时 $\exp(-705/2)\approx10^{-154}$，热核矩阵退化成单位阵、$L\equiv0$，$1/\mathrm{Tr}(G^\top LG)$ 被 `1/eps` 兜底成 $4.5\times10^{15}$ | METHOD §1.4、§5.4 |
| 3 | **乘性更新之间插硬二值化** | 变量被锁死在 $\{0,1\}$，且每列必有一个 1 | 代码每轮对 `y[i]`、`Y_c`、`Y_n` 做 MinMax+0.5（论文没写这一步），破坏"目标单调"的前提，$0$ 又成了吸收态 | METHOD §5.2（DHLI / I²VSLC 行） |
| 4 | **视图顺序敏感** | 换个视图排列，结果就变 | 只要方法里有 FFT 或按视图拼接的权重，切片顺序就改变结果。实测 TOCL 排名相关 0.456，UGRFS 0.988 | REPRO §1.4 |
| 5 | **伪标签"自证"** | 以为模型学到了标签结构，其实主要来自初始化 | 伪标签与模型联合优化时容易塌缩到初值或全 1。**验证方法是记录 $y^{(i)}$ 与随机初值的距离、逐轮熵，而不是只看指标**。本仓库没有专门量化这一条，但代码里 $y^{(i)}$ 每轮被硬二值化，使这种塌缩更难察觉 | METHOD §5.2；`python -m insight.run_demo M2 M4` |
| 6 | **指标口径不一致** | 复现结果"差 3 倍" | Coverage 是否除以 $L$、AP 是否在 1%–20% 上取平均、HL 是 micro 还是 macro —— 任一不同都会造成巨大差异 | 本笔记 §2.2；REPRO §1.3 |
| 7 | **随机种子泄漏** | 同参数同数据两次跑，结果不同 | 初始化用了全局 RNG（`np.random.rand`）而不是受种子控制的 `Generator`；仓库修过 TOCL 的这处 | REPRO §7.1；`alg/TOCL.py` L114–118 |

---

## 7. 上手路线（建议按这个节奏走）

### Day 1：把管道跑通，建立"数字感"

1. `.venv\Scripts\python.exe main.py --list` 和 `--check`；
2. 跑 `--alg DHLI --data emotions`，打开 `results/metrics_DHLI_emotions.csv`，**画出"AP vs 特征数"曲线**（自己用 matplotlib 画一遍，比看表格有用）；
3. 改 `--select-ratio 0.05` 再跑一次，观察曲线形状变化。

**目标**：明白"特征选择输出排序 → 取前 k → MLkNN → 指标"这条链，以及曲线为什么长这样。

### Day 2：读一篇论文 + 对着代码读公式

推荐从 **UGRFS**（`alg/UGRFS.py`，173 行）或 **DHLI**（230 行）开始，对照
`PAPER_METHOD_ANALYSIS.md` §3.2 / §3.4 的"变量清单 + 逐项解剖 + 代码对照"，
在代码里把每个变量和公式对上号。

**目标**：能回答 §4.6 的四个问题。

### Day 3–5：指标与协议

读 `EXPERIMENT_PROTOCOLS.md`，然后自己实现一遍 AP / Coverage / Ranking Loss（不要用 sklearn），
与 sklearn 的输出对比，再自己验证"Coverage 除以 $L$"这件事。

**目标**：以后看到任何论文表格，你都能判断它的口径是否可比。

### Week 2：复现 + 找改进点

1. 挑一个便宜数据集（`3sources`），把 `--alg all` 跑完，用 `paper_table.py` 生成对比表；
2. 读 `insight/FORMULA_TO_CODE.md` 与 `python -m insight.run_demo M1 ... M8`（8 个机制的可证伪验证）；
3. 从 `PAPER_METHOD_ANALYSIS.md` §6 的八条改进思路里挑一条，用 `insight/mechanisms.py` 的对应实现做实验。

**目标**：从"读懂"过渡到"能改"，并拥有一个自己验证过的小结论。

---

## 8. 学习资源

> 标 ⭐ 的是**优先读**；标 🔧 的是工具；标 📦 的是数据/代码。

### 8.1 综述（先读这两篇，建立全局观）

| 资源 | 为什么读 / 读哪部分 |
|---|---|
| ⭐ Zhang & Zhou, *A Review on Multi-Label Learning Algorithms*, TKDE 2014 —— [IEEE CSDL](https://www.computer.org/csdl/journal/tk/2014/08/06471714/13rRUwInvBt)（DOI 10.1109/TKDE.2013.39） | 多标签领域的"圣经级"综述。**读 §指标定义、§评价方法**：本仓库四个指标的原始定义与口径都来自这里。方法部分可跳读 |
| ⭐ Xu, Tao, Xu, *A Survey on Multi-view Learning* —— [arXiv:1304.5634](https://arxiv.org/abs/1304.5634) | 多视图学习的经典综述。**读 §2 一致性/互补性原则、§3 各类方法**。理解了"co-regularization / co-training / 共享子空间"三大家族，六篇论文的定位就清楚了 |
| Zhao et al., *Multi-view learning overview: Recent progress and new challenges*, Information Fusion 2017 | 比上一篇更新的多视图综述，看"deep multi-view"之前的版图（按标题+摘要粗读即可） |
| 软件学报 / 计算机学报上的**多视图学习**或**多标记特征选择**中文综述 | 中文术语对照用；挑一篇近 5 年的即可 |

### 8.2 奠基方法论文（读方法的"祖宗"，理解套路）

| 资源 | 为什么读 |
|---|---|
| ⭐ Nie et al., *Efficient and Robust Feature Selection via Joint $\ell_{2,1}$-Norms Minimization*, NIPS 2010 —— [PDF](https://proceedings.neurips.cc/paper/2010/file/09c6c3783b4a70054da74f2538ed47c6-Paper.pdf) | **$\ell_{2,1}$ + 行范数排序的源头**。看懂它，六篇的排序项就都懂了。短小精悍 |
| ⭐ Argyriou et al., *Convex Multi-task Feature Learning*, MLJ 2008 —— [DOI](https://dl.acm.org/doi/10.1007/s10994-007-5040-8)；NIPS 2006 版 [PDF](https://papers.NeurIPS.cc/paper_files/paper/2006/hash/0afa92fc0f8a9cf051bf2961b06ac56b-Abstract.html) | $\ell_{2,1}$ 用于"多任务共享特征"的源头；多标签 = 把"任务轴"换成"标签轴" |
| ⭐ Zhang et al., *Manifold Regularized Discriminative Feature Selection for Multi-Label Learning* (MDFS), Pattern Recognition 2019 —— [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302341) | **图正则 + 多标签稀疏回归**的组合范式，仓库六篇的直接前身之一。注意它是六篇的主要基线之一 |
| Zhang et al., *Multi-View Multi-Label Learning with Sparse Feature Selection for Image Annotation*, IEEE TMM 2020 —— [IEEE](https://ieeexplore.ieee.org/document/8960273) | 多视图多标签稀疏特征选择的代表作，"先拼接 + $\ell_{2,1}$"这一路的代表（GRAFS、I²VSLC 的参考文献里都引了它；它们批判的"两步法"另有其文） |
| Zhu et al., *Global and local multi-view multi-label learning*, Neurocomputing 371 (2020) 67–77 | 六篇反复点名的"两步法/全局视图拼接"前作（GRAFS 的 [9]、I²VSLC 的 [39]）。找到 PDF 后重点看它的 global/local 两类映射 |
| Lee & Seung, *Algorithms for Non-negative Matrix Factorization*, NIPS 2001（及 Ding et al. 2008 的 KKT 视角） | **乘性更新**的源头；读它才能理解 §4.4 那三行推导 |
| Cai et al., *Graph Regularized Nonnegative Matrix Factorization* (GNMF), TPAMI 2011 | **图正则 + 乘性更新**合在一起的范本；六篇的"图 + 乘性更新"就是这个组合 |

### 8.3 工具与代码

| 资源 | 用途 |
|---|---|
| 🔧 [scikit-feature](https://github.com/jundongl/scikit-feature)（ASU 特征选择库） | 六篇代码里 `skfeature.utility.construct_W` 就来自这里；想试各种 filter/embedded 方法是首选 |
| 🔧 [scikit-multilearn-ng](https://pypi.org/project/scikit-multilearn-ng/) | MLkNN 等多标签分类器。注意：**原版 `scikit-multilearn` 0.2.0 与 sklearn ≥1.0 不兼容**（`NearestNeighbors` 位置参数问题），本仓库 `main.py` 专门子类化修过；新项目建议直接用 ng 版 |
| 🔧 本仓库 `main.py` / `tune.py` / `paper_table.py` | 一条命令跑完"排序→MLkNN→四指标→与论文表格对比"的全链路，比自己搭评测省一周 |
| 🔧 `insight/mechanisms.py` + `run_demo.py` | 8 个核心机制的**最小可运行实现 + 可证伪验证**，是"公式→代码"最快的教材 |
| 📦 本仓库 `data/*.mat`（10 个数据集） | 已经整理好的 MVML 数据，省去自己找数据/对齐维度的痛苦 |
| 📦 Mulan 多标签数据集页（`mulan.sourceforge.net/datasets-mlc.html`） | 更广的多标签数据集（yeast、scene、emotions 等的原始出处）；sourceforge 年代久远，若打不开就用仓库里的 `.mat` |
| 📦 MIRFlickr / Corel5K / NUS-WIDE 官方页 | 图像多标签的原始数据；注意本仓库用的是它们的**预处理多视图版本**（DH/GIST/HH 等 100/300/512 维特征） |

### 8.4 书与课程

| 资源 | 用途 |
|---|---|
| 周志华《机器学习》**第 11 章 特征选择与稀疏学习** | 过滤式/包裹式/嵌入式、$\ell_1$ 正则、稀疏表示的清晰中文讲解。**§11.1 子集搜索与评价、§11.4 嵌入式选择**是本方向的直接背景 |
| 李航《统计学习方法》 | 优化与正则化的基础（凸性、KKT、近端算法）；查工具书用 |
| 任何一本凸优化（Boyd《Convex Optimization》第 5、§4.2 讲 ADMM 的部分） | 读 §4.4/§4.5 与 ADMM 类论文时的参考 |
| 你所在学校的"机器学习/模式识别"研究生课 | 把多标签/多视图当作专题来做课程报告，是最快进入的方式 |

### 8.5 本仓库已有的六篇论文与笔记（**最相关**）

| 资源 | 用途 |
|---|---|
| `pre-pdf/*.pdf`（六篇全文）+ `pre-pdf/paper-notes/*.md` | 逐篇题录、摘要、符号↔代码对照 |
| `analysis/PAPER_METHOD_ANALYSIS.md` | 逐篇讲透 + 代码偏差 + 八条改进思路（本文的"深水区续集"） |
| `analysis/EXPERIMENT_PROTOCOLS.md` | 六篇实验设置对比、哪些数不可比 |
| `analysis/REPRODUCE_TOCL_UGRFS.md` | 复现指南 + 实测耗时/内存 |
| `analysis/README_zh.md` | 运行器参数说明 |

---

## 9. 六篇论文的阅读顺序（附地图）

> 仓库里 §3 是按 **README 的论文顺序**（TOCL → UGRFS → EF²FS → DHLI → GRAFS → I²VSLC）排的；
> 但**第一次读建议按"由易到难 + 由源到流"的顺序**，见下表。

| 顺序 | 论文 | 为什么这个位置 | 关键收获 |
|---|---|---|---|
| 1 | **DHLI**（AAAI 2024，`alg/DHLI.py`） | 这条线**最早**的一篇，也是概念最干净的一篇：标签切三类 + 离散逻辑算子 | 学会"把噪声变成变量"的论证方式；四则逻辑算子的连续化技巧 |
| 2 | **I²VSLC**（Inf. Sci. 2024） | 与 DHLI 同期但更简单（没有逻辑算子），代码只有 193 行 | 视图特有伪标签 + 两层图正则；**最容易自己复现** |
| 3 | **GRAFS**（Inf. Sci. 2024） | 结构最"工程"：锚点 + 双路径分解 | 理解"全局视图是学出来的"这一转折；看懂软一致性约束的必要性 |
| 4 | **UGRFS**（AAAI 2025） | 骨架与 GRAFS 很像但多了一个"样本置信度" | 理解 A/B 分界的真实判据（新数学对象 + 领域动机） |
| 5 | **EF²FS**（PR 2025） | 唯一给了单调性论证的一篇 | 学习怎么用辅助函数证明收敛；看"融合进损失"的设计 |
| 6 | **TOCL**（ACM MM 2025） | 最难的放最后：张量 t-SVD + 加权核范数 + 双路映射 | 新工具引入型创新怎么写；也是**坑最多**的一篇（§5.2、§5.6） |

**配套读法**：每读一篇，打开 `PAPER_METHOD_ANALYSIS.md` 的对应 §3.x，按
"一句话 → 变量清单 → 逐项解剖 → 更新式来源 → 代码对照"的顺序过一遍，
再把 `alg/<名字>.py` 打开对照。六篇读完，你手里就有了这一支技术的完整拼图。

**然后读 §2 与 §4**（四轴演进图 + 时间线血缘），把六篇串成一条线：
"标签拓扑/视图拓扑/损失拓扑/优化拓扑"四条轴上的演进，以及"前一篇的软肋就是后一篇的动机"。

---

## 10. 术语中英对照

| 中文 | 英文 | 备注 |
|---|---|---|
| 多标签学习 | multi-label learning | 输出是标签集合 |
| 多视图学习 | multi-view learning | 同一批样本的多种特征表示 |
| 多视图多标签 | multi-view multi-label (MVML) | 本文档主角 |
| 特征选择 | feature selection | filter / wrapper / embedded 三类 |
| 嵌入式方法 | embedded method | 选择与学习同时进行（六篇属此类） |
| 行稀疏 / 组稀疏 | row sparsity / group sparsity | 由 $\ell_{2,1}$ 诱导 |
| 流形正则 / 图正则 | manifold / graph regularization | $\mathrm{Tr}(Y^\top LY)$ |
| 图拉普拉斯 | graph Laplacian | $L=A-S$ |
| 视图特有标签 | view-specific label | 本组的关键变量 $y^{(i)}$ |
| 伪标签 | pseudo-label | 用模型自己生成的监督信号 |
| 一致性与互补性 | consistency & complementarity | 多视图的两大信息类型 |
| 乘性更新 | multiplicative update | 非负优化的一步 KKT |
| 辅助函数 | auxiliary function | 证明单调性的工具（MM 算法） |
| 张量奇异值分解 | t-SVD | 沿 tube 方向 FFT 后再做矩阵 SVD |
| 张量核范数 / 管秩 | tensor nuclear norm / tubal rank | TOCL 的核心工具 |
| 锚点 | anchor | 用少量代表点表达全部样本 |
| 平均精度 / 覆盖率 / 汉明损失 / 排序损失 | AP / Coverage / Hamming Loss / Ranking Loss | 四个论文指标 |
| 标签基数 / 标签密度 | label cardinality / label density | 每个样本平均标签数 / 除以 $L$ |
| 五折交叉验证 | 5-fold cross validation | 默认协议 |
| 权衡系数 / 平衡参数 | trade-off / balance parameters | $\alpha,\beta,\gamma,\delta$；**逐篇含义不同**，见 §5.3 |
| 主拟合项 | main fitting / data term | 模板里系数固定为 1 的那一项 |

---

## 附：读完这份笔记的自测题

试着不看文档回答（答案都能在本文或 `PAPER_METHOD_ANALYSIS.md` 里找到）：

1. 为什么特征选择的排序量是 $\|W_{(j)}\|_2$ 而不是 $|W_{pq}|$？
2. $\mathrm{Tr}(Y^\top LY)$ 为什么等价于"图上相邻样本的表示要相近"？
3. 乘性更新的分子和分母分别对应梯度的哪一部分？为什么 $0$ 是吸收态？
4. Coverage 报出来是 0.42 还是 13.6，取决于什么？
5. 六篇论文里，哪一篇引入了"视图特有标签"？哪一篇引入了"样本置信度"？哪一篇引入了张量核范数？
6. "目标值相对变化率"这条曲线为什么不能作为收敛证据？
7. 热核下溢会让某个视图发生什么？根因是哪个参数？
8. 5 视图数据集的视图顺序问题会怎样影响 TOCL 的结果？
9. 为什么"把一致/互补分开提取"会被后来者批评？
10. 如果让你在 METHOD §6 的八条改进思路里挑一条做实验，你会挑哪条，验证什么？
11. 模板（§0.2.1）里哪一项的系数被固定为 1？为什么必须固定？
12. $\delta$ 在六篇里分别管什么？哪一篇是例外？
13. UGRFS 的 $\gamma$ 和 I²VSLC 的 $\delta$ 是同一件事吗？

> 能把 1–8 讲清楚，说明你已经具备读**任何** MVML-FS 论文的基础；
> 能对 9–13 给出自己的判断，就可以开始找选题了。
