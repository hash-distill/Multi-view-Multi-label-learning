# 六篇多视图多标签特征选择论文的方法论分析

> **分析对象**
>
> | 简称 | 论文 | 发表处 | CCF | 本仓库代码 |
> |---|---|---|---|---|
> | **TOCL** | Tensor-based Opposing yet Complementary Learning | ACM MM 2025, pp.1822-1831 | **A** | `alg/TOCL.py` |
> | **UGRFS** | Uncertainty-Aware Global-View Reconstruction | AAAI 2025, 39(16) | **A** | `alg/UGRFS.py` |
> | **EF²FS** | Embedded feature fusion | Pattern Recognition 157 (2025) 110888 | **B** | `alg/EF2FS.py` |
> | **DHLI** | Double-Layer Hybrid-Label Identification | AAAI 2024, 38(11):12295-12303 | **A** | `alg/DHLI.py` |
> | **GRAFS** | Anchor-guided global view reconstruction | Inf. Sci. 679 (2024) 121124 | **B** | `alg/GRAFS.py` |
> | **I²VSLC** | Exploring view-specific label relationships | Inf. Sci. 681 (2024) 121215 | **B** | `alg/I2VSLC.py` |
>
> 全部出自同一研究组（Pingting Hao 等，东北师范大学 / 吉林大学），构成一条
> **连续演进的技术路线**，而非六篇独立工作。这是理解其创新点的关键前提。
>
> **配套材料**
> * `insight/mechanisms.py` —— 8 个核心机制的最小可运行实现（逐行对应公式）
> * `insight/run_demo.py` —— 每个机制的可证伪性验证（`python -m insight.run_demo`）
> * `insight/FORMULA_TO_CODE.md` —— 公式 ↔ 代码逐条对照与偏差表
> * `CODE_WALKTHROUGH.md` —— 六份 `alg/*.py` 按**代码先后顺序**的逐块解读（本文 §3 各篇"代码对照"的展开版）
>
> **重要取证说明**：仓库 `_pdftext/` 下的纯文本是 pdfminer 流式抽取，**公式编号与
> 多列表格存在系统性错乱**（例如 DHLI 的式(2) 被标成(5)；Table 3 的 8 列被交错成
> 8+4+4 段，会造出"某基线 HL 差 20 倍"的假象）。本文档的数字已用 PyMuPDF
> 按坐标重排 PDF 核对。凡引用具体数值处均以此为据。
>
> **排序约定**：本文档所有"逐篇"内容（上面这张分析对象表、§3 逐篇拆解、§5.2 逐篇偏差、
> §5.3 表）统一按仓库 `README.md` 的 Publications 顺序排列 ——
> **TOCL → UGRFS → EF²FS → DHLI → GRAFS → I²VSLC**（不是发表时间顺序）。
> 只有 §2 与 §4 例外：它们讨论的正是**演进脉络与时间线**，因此保持"旧 → 新"的时间序；
> §6 的机制编号 M1–M8 被 `insight/` 下的脚本与文档引用，保持稳定不重排。

---

## 目录

1. [统一记号与问题形式化](#1-统一记号与问题形式化)
2. [这一线工作的四个技术轴与演进脉络](#2-这一线工作的四个技术轴与演进脉络)
3. [逐篇拆解](#3-逐篇拆解)
   - [3.1 TOCL (ACM MM 2025)](#31-tocl--张量对立互补学习acm-mm-2025)
   - [3.2 UGRFS (AAAI 2025)](#32-ugrfs--不确定度感知的全局视图重构aaai-2025)
   - [3.3 EF²FS (PR 2025)](#33-ef²fs--嵌入式特征融合pattern-recognition-2025)
   - [3.4 DHLI (AAAI 2024)](#34-dhli--双层级混合标签识别aaai-2024)
   - [3.5 GRAFS (Inf. Sci. 2024)](#35-grafs--锚点引导的全局视图重构inf-sci-2024)
   - [3.6 I²VSLC (Inf. Sci. 2024)](#36-i²vslc--视图特有标签关系inf-sci-2024)
4. [论文之间的"问题→改进"闭环](#4-论文之间的问题改进闭环)
5. [代码与论文的偏差总表（含新发现的缺陷）](#5-代码与论文的偏差总表含新发现的缺陷)
6. [八条可验证的改进思路：公式 + 代码](#6-八条可验证的改进思路公式--代码)
7. [若要在此基础上再发一篇：定位建议](#7-若要在此基础上再发一篇定位建议)

---

## 1. 统一记号与问题形式化

### 1.1 记号

| 符号 | 含义 | 维度 |
|---|---|---|
| $n$ | 样本数 | 标量 |
| $V$ | 视图数 | 标量 |
| $d^{(i)}$ | 第 $i$ 个视图的特征数；$d=\sum_i d^{(i)}$ | 标量 |
| $l$（部分论文用 $c$） | 标签数 | 标量 |
| $X^{(i)}$ | 第 $i$ 个视图特征矩阵 | $n\times d^{(i)}$ |
| $Y$ | 观测标签矩阵 | $\{0,1\}^{n\times l}$ |
| $W^{(i)}$ | 第 $i$ 个视图的特征权重 | $d^{(i)}\times l$ |
| $W=\bigoplus_i W^{(i)}$ | 纵向拼接的全局特征权重 | $d\times l$ |
| $y^{(i)}$ | 视图特有（伪）标签 | $n\times l$ |
| $L^{(i)}$ | 图的拉普拉斯矩阵 | $n\times n$ |
| $\|\cdot\|_{2,1}$ | 行组稀疏范数 $\sum_p\sqrt{\sum_q(\cdot)_{pq}^2}$ | — |

### 1.2 六篇论文的共同问题骨架

> **本节起是"共性总纲"**：§3 的六篇拆解**只讲各自的方法与创新点**；凡是六篇共用的东西
> —— 任务设定、目标函数模板与四个权衡系数（§1.2）、共同的方法论软肋（§1.4）、
> 三件共用数学工具（§1.5）、读一篇目标函数的固定套路（§1.6）—— 都集中在 §1 讲完。

**任务设定。** 多视图多标签特征选择（MVML-FS）的输入输出是：

| | 形式 | 含义 |
|---|---|---|
| 输入 | $X^{(i)}\in\mathbb{R}^{n\times d^{(i)}},\ i=1..V$ | 同一批 $n$ 个样本的第 $i$ 个视图（一张图的 RGB 直方图 / GIST / 词袋；一篇文章的标题词袋 / 正文 TF-IDF） |
| 输入 | $Y\in\{0,1\}^{n\times l}$ | 观测多标签矩阵，$Y_{pj}=1$ 表示样本 $p$ 带第 $j$ 个标签 |
| 输出 | 长度 $d=\sum_i d^{(i)}$ 的**特征排序** | 只排序、不分类；下游取前 $k$ 个特征训分类器，再算 AP / Coverage / HL / RL |

困难可归为三条，读每篇时拿它们去对：**（i）共性 vs 互补**（各视图既冗余又互补，怎么把两者分开是 §2 轴 II 的全部内容）；
**（ii）标签噪声**（$\{0,1\}$ 里既有假阳也有假阴，而去噪过头就没监督信号了 —— DHLI 把这条误差传导链讲得最清楚）；
**（iii）排序目标 vs 连续优化**（真正的目标是选子集、NP-hard，所有方法都把它松弛成连续稀疏优化）。
（慢速展开版见 `MVML_PRIMER.md` §0.3。）

所有六篇都在解同一个**嵌入式（embedded）多视图多标签特征选择**问题。其骨架是：

$$
\min_{\substack{W^{(i)},\,y^{(i)}\\ \text{（及若干辅助变量）}}}
\;\underbrace{\sum_{i=1}^{V}\mathcal{L}\big(X^{(i)}W^{(i)},\,y^{(i)}\big)}_{\text{(a) 映射拟合}}
\;+\;\underbrace{\lambda\,\mathcal{R}\big(W^{(1..V)}\big)}_{\text{(b) 稀疏 → 排序}}
\;+\;\underbrace{\sum_k \gamma_k\,\mathcal{C}_k\big(\cdot\big)}_{\text{(c) 结构约束}}
\tag{1.1}
$$

特征重要性 = $W$ 的**逐行 2-范数** $\|W_{(j)}\|_2$，降序取前 $k$ 个。

**六篇的差别全部在 (a) 的目标变量和 (c) 的结构约束上**。把它列成一张表，演进脉络就一目了然：

| 论文 | (a) 拟合到什么 | (c) 结构约束的核心 | 一句话增量 |
|---|---|---|---|
| TOCL | $PX^{(i)}A^{(i)}W$ 与非线性映射 | **张量**核范数 + 分层标签相关 | 跨视图高阶关系进张量 |
| UGRFS | $\mathrm{diag}(C^{(i)})X^{(i)}W^{(i)}$ | 样本置信度 + 全局视图双向拆解 | 样本可信度从常数 1 变变量 |
| EF²FS | $X^f A$（融合后嵌入） | 融合↔选择闭环 + 全局/局部权重一致 | 融合不再是预处理 |
| DHLI | $Y_c$ 与 $y^{(i)}$ **两路** | 标签三类完全切分 + 逻辑（AND/OR/XOR）正则 | 把"噪声"变成可优化变量 |
| GRAFS | $X^f W$（重构全局视图） | 锚点双路径分解 + 共享上下文 | 全局视图从"假设"变"变量" |
| I²VSLC | $y^{(i)}$ | 视图内标签流形 + 视图间共识 + 动态标签相关 | 把"标签关系"按视图拆开 |

**骨架的三个槽位与四个权衡系数。** 式(1.1) 就是这一支的"填空模板"：

$$
\min\;\underbrace{\mathcal{L}_{\text{fit}}}_{\text{主拟合项：系数固定为 }1}
\;+\;\alpha\,\mathcal{C}_1\;+\;\beta\,\mathcal{C}_2\;+\;\gamma\,\mathcal{C}_3
\;+\;\delta\underbrace{\|W\|_{2,1}}_{\text{稀疏 → 排序}}
\tag{1.2}
$$

* **主拟合项的系数固定为 1**，否则模型不可辨识（整体乘 2 不改变最优解的结构）⇒ **自由系数恰好 4 个**。
  这就是六篇都写 "four trade-off parameters $\alpha,\beta,\gamma,\delta$" 的来源，也是它们不约而同地
  在 $\{10^{-3},\dots,10^{3}\}$（7 个取值）上**逐个参数扫描**的原因（见 `EXPERIMENT_PROTOCOLS.md`）。
* **三个结构槽位里放什么 = 这篇论文的全部创新**。$\mathcal{C}_1,\mathcal{C}_2,\mathcal{C}_3$ 的内容可归为三类：
  **图/流形平滑**（§1.5.1）、**一致性/重构**（全局视图 ⇄ 各视图、双路映射互推）、**标签结构**（标签相关、hybrid labels、噪声项）。
* **六篇里有五篇的 $\delta$ 槽都是行稀疏**，唯一例外是 DHLI（它的 $\delta$ 管视图互斥，稀疏放在 $\gamma$）。
* ⚠️ **四个字母在不同论文里含义不同**（同一个 $\alpha$ 管的项可以完全不一样）：逐篇对照表见
  `MVML_PRIMER.md` §5.3。**不要记字母，记"这一项在管什么"。**

### 1.3 CCF 分层的真实判据

分析这六篇之前必须先讲清楚"为什么它上了 A 而它只上 B"。从审稿人视角，这一线工作的接收判据并非"SOTA 幅度"，而是三条：

| 判据 | 含义 | 达标者 |
|---|---|---|
| **J1 是否提出了新的问题设定或新的数学对象** | 不是"更好的超参/更强的正则"，而是引入了**此前不存在于本领域的量** | DHLI（hybrid labels）、UGRFS（样本置信度）、TOCL（张量低秩） |
| **J2 是否把某个人工先验变成了可学变量** | "predefined → learned" 是这一线最强的创新叙述 | GRAFS（$X^f$）、EF²FS（$G$/$c_i$）、UGRFS（$C^{(i)}$）、I²VSLC（$y^{(i)}$） |
| **J3 实验矩阵是否饱和** | 6 数据集 × ≥6 基线 × 4 指标 × 5 折 × 1–20% 特征比例 + 参数 + 收敛 + 消融 | 六篇全部满足 |

**关键观察**：六篇的**性能优势幅度都很小**（多数落在标准差内，见 §3 各篇），所以 J3 是**入场券**而非加分项，真正决定档次的是 J1 与 J2。这也解释了为什么 AAAI 的两篇（DHLI、UGRFS）和 ACM MM 的一篇（TOCL）都是在**引入新数学对象**，而三篇期刊论文（I²VSLC、GRAFS、EF²FS）主要是**把先验变变量 + 组合式增量**。

### 1.4 六篇共同的方法论特征（也是共同的软肋）

| 特征 | 证据 |
|---|---|
| **全部使用乘性更新（multiplicative update rules）** | 六篇正文均出现该词；模板为 $Z\leftarrow Z\circ\frac{[\nabla\Theta]^-}{[\nabla\Theta]^+}$ |
| **全部声称收敛，但全部没有定理** | 六篇 PDF 中 `Theorem`/`Lemma`/`Proof`/`Auxiliary`/`monotone` 的命中数**均为 0**；唯一的收敛证据是目标值相对变化率曲线 |
| **DHLI 的措辞最强、证据最弱** | 贡献点写 "with **proven** convergence"，但 9 页正文无任何定理、引理或附录。同页的收敛段还误写成 "the convergence of **LGCM**"（他文残留） |
| **目标值曲线的纵轴选得可疑** | 六篇统一画 $(z^{t-1}-z^t)/z^{t-1}$。**震荡序列的相邻差比单调下降序列更大**，因此震荡的迭代反而能画出"下降更快"的曲线（§6.7 有实证） |
| **非光滑项一律用辅助对角阵松弛** | $\|W\|_{2,1}=2\mathrm{Tr}(W^\top EW)$、$e_{ii}=1/(2\|W_i\|_2)$，六篇同构 |
| **图拉普拉斯一律 `construct_W(knn, heat_kernel, t=1.0)` 一次算死** | 这是 §5、§6 里最重要的可攻击点 |

> **⚠️ 一个贯穿全篇的实证发现**：`t=1.0` 这个固定带宽在真实数据上会**热核下溢**。
> 实测（`python -m insight.run_demo M3 M5 M8`）：当某视图的中位平方距离为
> $705$ 时，$\exp(-705/2)\approx 10^{-154}$ 已接近 float64 下界，热核矩阵退化为
> 单位阵、$L\equiv 0$。后果是 $1/\mathrm{Tr}(G^\top LG)$ 被 `1/eps` 兜底成
> $\sim 4.5\times10^{15}$，**该视图独吞全部权重**。这条缺陷同时出现在
> UGRFS / GRAFS / EF²FS / I²VSLC 四篇的**同一行公式**上。

**外加四条跨篇缺陷的结论（证据分别见括号处，§3 各篇不再重复）**：

1. **四篇共用同一个有缺陷的视图权重公式** $v_i\propto 1/\mathrm{Tr}(G^\top LG)$：对 $G$ 的尺度是**二次**的、与标签**无关**、热核下溢时会**塌缩到单一视图**（§5.4）。
2. **$\gamma$ 与 $t$ 的量纲在各篇之间不一致**，因此"同一套 $10^{-3}\sim10^{3}$ 网格"实际上并不可比（§5.5）。
3. **声称的范数 ≠ 实际求解的范数**：TOCL 的加权 TNN 权重 $w_i=C/(\sigma_i+\varepsilon)$ 与它给出的闭式解（Eq.24）**不自洽** —— "先有代码、后有推导"的典型痕迹（§5.6）。
4. **写死的结构参数**：TOCL 的 `delta`/`aaa`/`rho`、GRAFS 的 $k_1{=}10$ / $kk{=}20$、EF²FS 的 `V_dim{=}30` 论文全都没报 ⇒ 复现空洞（§5.2）。
   另外 DHLI / I²VSLC / TOCL 的代码都在**乘性更新之间插入硬二值化**，直接破坏"目标单调"的前提（§5.2）。

> **跨篇缺陷的完整证据在 §5.4–§5.7**：§5.4 视图权重公式（四篇共用同一行）、§5.5 $\gamma$/$t$ 的量纲、
> §5.6 TOCL 加权 TNN 的内部不自洽、§5.7 目标值曲线不可作为收敛证据；逐篇的代码级偏差在 §5.2。

### 1.5 六篇共用的三件数学工具

> 只列"读 §3 时一定会用到"的部分；每个工具的慢速讲解见 `MVML_PRIMER.md` §4.1–§4.4。

#### 1.5.1 图 / 流形正则：$\mathrm{Tr}(F^\top LF)=\tfrac12\sum_{p,q}w_{pq}\lVert f_p-f_q\rVert^2$

给定样本间的相似度图（六篇都用 `construct_W(knn, heat_kernel, t=1.0)` 建一次就固定），
$D=\mathrm{diag}(\sum_q w_{pq})$、$L=D-W$。对任意信号矩阵 $F$（每列一条信号）：

$$
\frac12\sum_{p,q}w_{pq}\big\|f_p-f_q\big\|^2=\mathrm{Tr}(F^\top LF),
\qquad L\mathbf 1=0,\quad L\succeq0
$$

（左式展开后用 $D,W$ 的定义合并即得。）**六篇只是换了"图建在哪、正则作用在谁身上"的组合**：

| 论文 | 图建在 | 正则作用在 | 代码 |
|---|---|---|---|
| I²VSLC | 视图特征 | 视图特有标签 $y^{(i)}$ | `Lx_lst[i]` |
| UGRFS | **标签** | 全局视图 $D$ | `Ly` |
| EF²FS | 视图特征 | 隐标签 $G$ | `Lx_lst[i]` |
| GRAFS | 论文写视图特征，**代码建的是标签图** | 视图权重 $v_i$ | `Ly`（§5.2 GRAFS 行） |

#### 1.5.2 行稀疏：$\ell_{2,1}$ 既做正则、也定义排序量

$\|W\|_{2,1}=\sum_p\|W_{(p)}\|_2$ 的"组"是**行** = 一个特征（对所有标签共享），其近端解会把**整行**压成 0。
它因此同时承担两件事：**正则**（把没用的特征整行清零）与**排序**（$\|W_{(j)}\|_2$ 降序即特征重要性）。
非光滑项六篇一律用辅助对角阵松弛：$\|W\|_{2,1}=2\mathrm{Tr}(W^\top EW)$、$e_{ii}=1/(2\|W_i\|_2)$，
于是"迭代重加权"实现行稀疏 —— **当前行范数越小，惩罚越大**。

#### 1.5.3 乘性更新：非负假设下的一步 KKT

六篇都写同一种更新：固定其余变量，然后 $Z\leftarrow Z\circ[\nabla\Theta]^-/[\nabla\Theta]^+$。
它不是启发式，而是**非负约束下的一步 KKT**：

1. 所有数据和变量非负（六篇的数据是直方图/GIST/词袋这类非负特征；**若含负值，分子可能为负、更新会破坏非负假设甚至发散**）；
2. 把子问题梯度按符号拆开：$\nabla f(Z)=\nabla^+f-\nabla^-f$，两项逐元素非负；
3. 对 $\min f(Z)\ \mathrm{s.t.}\ Z\ge0$ 写 KKT：$Z\circ\nabla f(Z)=0$，即 $Z\circ(\nabla^+-\nabla^-)=0$，
   于是 $Z=Z\circ\nabla^-/\,\nabla^+$。

**读更新式的规则**：分子里的加项 = 让该变量增大的力；分母里的加项 = 让它减小的力。
三个必须知道的后果：

* **没有步长**：不用选学习率，代价是每步尺度完全由数据决定；
* **$0$ 是吸收态**：某元素一旦被更新为 0 就再也出不来（这正是各篇都要加 `eps` 兜底的原因）；
* **收敛性无保证**：乘性更新只对能写出辅助函数的问题才单调下降，而六篇**都没给定理**（§5.7、§6.7）。

### 1.6 怎么读一篇的目标函数（三步 + 三个自检）

**三步**（§3 每篇的"逐项解剖"都是按这个顺序写的）：

1. 找**系数为 1 的主拟合项** ⇒ 定下"谁拟合谁"（§2 的"轴 III：损失拓扑"讲的就是这条轴）；
2. 把带 $\alpha/\beta/\gamma$ 的三项**各自命名** ⇒ 这三项就是这篇文章的创新点；
3. 找 $\delta\|W\|_{2,1}$ ⇒ 确认排序量确实是 $W$ 的行范数。

**三个自检**：

| 自检 | 做什么 | 作用 |
|---|---|---|
| **变量清单** | 列出每个符号：已知还是待学、维度、代码里的名字 | 记号不理顺，后面全乱 |
| **逐项解剖** | 每一项在约束谁：拟合 / 稀疏 / 图平滑 / 一致性 / 标签结构 | 分清"地基"与"创新" |
| **维度自检** | 把每个乘积的维度写出来验一遍 | 一眼看出哪里理解错了 |

---

## 2. 这一线工作的四个技术轴与演进脉络

六篇看似分散，实际沿四条轴演化。理解这四条轴，就理解了这个研究组的"写作方法论"。

### 轴 I：标签拓扑 —— 标签之间的关系怎么建

```
   共享一个 Y                     y⁽ⁱ⁾ 按视图拆分              Y 拆成三类 + 逻辑约束
   (lrMMC, M2LD)                  (MVLD, I²VSLC)               (DHLI, TOCL)
        │                              │                            │
        │  问题：忽略视图互补           │  问题：噪声仍在，且         │  DHLI 引入
        │                              │  降维不足以指明滤噪方向      │  hybrid labels
        ▼                              ▼                            ▼
   I²VSLC 用 Φ(·) 同时挂 intra/inter   DHLI 把噪声变成 Y_n          TOCL 再把标签相关
   → Inf. Sci. 2024                   → AAAI 2024                 升到"分层 + 张量"
                                                                  → ACM MM 2025
```

**关键转折**：DHLI 的贡献不是"更好的去噪"，而是把去噪从**隐式**（低秩、流形）变成**显式**（判定一个 $Y_n$ 变量）。TOCL 沿用这一设定并加上"第一层/第二层标签相关"的定义（Definition 3/4）。

### 轴 II：视图拓扑 —— 视图之间怎么融合

```
   直接拼接 X = [X⁽¹⁾,…,X⁽ⱽ⁾]      固定权重加权 ΣᵥᵢX⁽ⁱ⁾          重构出 X^f（学出来的）
   (多数基线)                       (UGRFS/GRAFS/EF²FS 的形式)     (GRAFS 提出)
        │                              │                            │
        │ 问题：忽略视图关系            │ 问题：vᵢ 一次算死、         │  问题：X^f 与标签
        │                              │ 尺度敏感（§1.4）            │  的关系仍是单层
        ▼                              ▼                            ▼
   GRAFS 让 X^f 成为待解变量        UGRFS 让 vᵢ 随 G 演化         EF²FS 把融合塞进损失
   → Inf. Sci. 2024                 → AAAI 2025                  → PR 2025
```

### 轴 III：损失拓扑 —— "谁拟合谁"

这是最容易被忽略、但区分度最高的一条轴：

| 论文 | 损失形式 | "谁拟合谁" |
|---|---|---|
| I²VSLC / DHLI | $\|X^{(i)}W^{(i)}-y^{(i)}\|_F^2$ | 特征 → 标签（单层） |
| GRAFS | $\|X^fW-Y\|_F^2$ | **重构视图** → 标签（两层） |
| EF²FS | $\|X^fA-G\|^2+\alpha\|Y-GB^\top\|^2+\beta\Sigma\|X^{(i)}w^{(i)}-G\|^2$ | 三方**共同**拟合 $G$（多边） |
| UGRFS | $\|\mathrm{diag}(C)XW-Y\|^2+\beta\Sigma\|D^{(i)}-\mathrm{diag}(C)X^{(i)}\|^2$ | 双向（$D$ 与 $X$ 互推） |
| TOCL | $\|PX^{(i)}A^{(i)}W-y_t^{(i)}\|^2+\alpha\|PXW-\sigma(XV)U\|^2$ | 线性 + **非线性**两路并行 |

**演进方向 = 从"单边拟合"走向"多边互推"，最终走向"对立互补"**（TOCL 的标题即此）。

### 轴 IV：优化拓扑 —— 怎么解

```
   乘性更新（全部六篇）
        │
        ├─ 优点：无步长超参、保持非负、实现简单
        └─ 缺点：无收敛保证；且实测**震荡**（§6.7）
                │
                ▼
   最新工作 THBFS（本仓库仅有宣传材料）已明确改用 **ALM + ADMM**
   → 说明研究组自己也意识到这条轴需要升级
```

---

## 3. 逐篇拆解

顺序：**TOCL → UGRFS → EF²FS → DHLI → GRAFS → I²VSLC**（即仓库 `README.md` 的论文顺序）。

**六篇共用的东西已在 §1 讲完**（任务设定与模板 §1.2、软肋 §1.4、共用数学工具 §1.5、读法 §1.6）；
本节每篇**只讲它自己的东西**，统一按：
**① 它认为前人错在哪 → ② 它的目标函数（变量清单 + 逐项解剖 + 维度自检）→ ③ 它的求解与更新式
→ ④ 它的创新点归因 → ⑤ 它的实验证据 → ⑥ 它的代码对照（公式 ↔ `alg/*.py`）**。

> **阅读提示**：先按 §1.6 的三步把目标函数拆成"主拟合项 + 三个结构槽 + 稀疏槽"，
> 再回答**"它额外引入了哪个可学变量？这个变量被哪几项牵着走？最后排序用哪个量？"**
> —— 这几问决定了这篇论文的贡献，其余细节基本都属于实现。

### 3.1 TOCL — 张量对立互补学习（ACM MM 2025）

> **一句话**：把每个视图各自的标签关联矩阵 $y_t^{(i)}y_t^{(i)\top}$ 堆成一个**三阶张量**，
> 用**加权张量核范数**把它压成低秩（= 强制各视图共享同一套标签相关结构，顺带把标签去噪），
> 同时用**一路线性映射 + 一路 sigmoid 非线性映射**去逼近"视图特有标签"。

**TOCL 的抓手**：前面几篇都已经意识到"每个视图应当有自己的标签 $y_t^{(i)}$"，但它们对
$y_t^{(i)}$ 的约束都是**逐视图独立**的（各自平滑、各自跟共识对齐），丢掉了"**视图与视图之间**"这一层。
TOCL 把 $V$ 个视图的标签关联矩阵堆成三阶张量，用**张量低秩**一次性刻画跨视图的高阶一致性；
同时指出前人只用了**一种**映射（要么线性要么非线性），而这两者其实是互补的 —— 这就是标题里的
"opposing yet complementary"。

#### 3.1.1 它认为前人错在哪

论文把已有工作分成两条线，各指出问题：

| 线索 | 原文 | 中译 | 问题 |
|---|---|---|---|
| 用观测标签（假设各视图标签一致） | "it is assumed that the standard label for each view is **consistent**" | 假设各视图标签与共享参考标签对齐 | 忽略视图特有部分 |
| 用观测标签作为所有视图的标签（不识别） | "the label for each view is **not fully identified**… this imprecise mapping across views often degrades feature selection performance" | 各视图标签未被完全识别 | 跨视图映射不精确 ⇒ 降低特征选择性能 |
| 用视图特有标签 | "some researches utilize a **two-stage** procedure that may experience performance degradation if the supervisory information derived from the first stage is **inaccurate**" | 两阶段：第一阶段监督信息不准会传导 | 误差在阶段间传播 |
| 联合模型 | "**prior studies often overlook different types of the mapping and higher-order constraints among view-specific labels**" | 忽略了**映射的类型差异**与**视图特有标签间的高阶约束** | **本文的核心定位** |

**最后一条是 TOCL 的核心定位**：前人都只用**单一种类**的映射（线性或非线性），
也都没有对"视图特有标签之间的关系"施加**高阶（张量级）**约束。

#### 3.1.2 目标函数

**四组件**（论文 Figure 1 与摘要）：

1. **局部线性映射 + 实例级重要性**：

$$
\mathcal{L}_1=\sum_{i=1}^{V}\big\|PX^{(i)}A^{(i)}W-y_t^{(i)}\big\|_F^2
\tag{3.1}
$$

其中 $U^{(i)}=A^{(i)}W$ 用辅助矩阵 $A^{(i)}$ 把**全局权重** $W$ 投影成**逐视图权重**
（论文 Figure 2 的核心技巧：$A^{(i)}$ 是把 $W$ 的对应行取出的选择器）。

2. **全局非线性映射**（逻辑回归式）：

$$
\mathcal{P}_1=\alpha\Big\|PXW-\underbrace{\frac{I}{I+e^{-XV}}}_{\text{记作 }F}U\Big\|_F^2
+\|U\|_F^2+\|V\|_F^2
\tag{3.2}
$$

3. **张量核范数约束**（跨视图高阶关系）：

把各视图的标签关联矩阵堆成三阶张量 $T\in\mathbb{R}^{n\times V\times n}$，
$T^{(i)}=y_t^{(i)}y_t^{(i)\top}$：

$$
\mathcal{L}_2=\|Z-T\|_F^2+\|T\|_{w,*}
\tag{3.3}
$$

**加权**张量核范数（Eq.22–25）：

$$
\|T\|_{w,*}=\frac1n\sum_{j=1}^{n}\big\|\bar T(:,:,j)\big\|_{w,*},
\qquad
w_i^{(j)}=\frac{C}{\bar S(i,i,j)+\varepsilon}
\tag{3.4}
$$

奇异值闭式解（Eq.24）：

$$
\bar S(i,i,j)=\frac{c_1+\sqrt{c_2}}{2},\quad
c_1=\bar S-\epsilon,\quad c_2=(\bar S-\epsilon)^2-4(C-\epsilon\bar S)
\tag{3.5}
$$

4. **离散标签相关**（分辨 "一致 / 噪声 / 特有"）：

$$
\mathcal{P}_2=\gamma\big\|Y_{all}\otimes Y_n\ominus Y\big\|_F^2+\gamma\|Y_n\|_F^2
\tag{3.6}
$$

**完整目标**（式 14）：

$$
\min\;\underbrace{\sum_i\big\|PX^{(i)}A^{(i)}W-y_t^{(i)}\big\|_F^2+\|Z-T\|_F^2+\tau\|T\|_*}_{\text{实例级映射 + 张量约束}}
+\underbrace{\alpha\big\|PXW-FU\big\|_F^2+\|U\|_F^2+\|V\|_F^2}_{\text{全局非线性映射}}
+\beta\|P\|_F^2+\gamma\mathcal{P}_2+\delta\|W\|_{2,1}
\tag{3.7}
$$

**变量清单：谁是已知量，谁是要学的量**

| 符号 | 维度 | 角色 | 代码里的名字 / 初始化 |
|---|---|---|---|
| $X^{(i)}$（拼接为 $X$） | $n\times d^{(i)}$（$n\times d$） | 已知特征 | `x[i]` / `X` |
| $Y$ | $n\times l$ | 已知观测标签 | `Y` |
| $W$ | $d\times l$ | **要学**：全局特征权重 = 最终排序依据 | `W`，$U[0,1)$ 随机 |
| $A^{(i)}$ | $d^{(i)}\times d$ | **不更新**：把 $W$ 的对应行块"取出来"的 0/1 选择矩阵 | `d[i]`，循环外只构造一次 |
| $P$ | $n\times n$ | **要学**：两路映射的耦合器（论文称"实例级重要性"） | `P`，随机 |
| $y_t^{(i)}$ | $n\times l$ | **要学**：第 $i$ 个视图的视图特有标签 | `y[i]`，随机二值 |
| $U$ | $k\times l$，$k=\lfloor 0.8l\rfloor$ | **要学**：隐标签关联矩阵 | `U`，随机 |
| $V$ | $d\times k$ | **要学**：非线性映射的投影系数 | `V`，随机 |
| $T$ / $Z$ | $n\times V\times n$ | **要学**：标签关系张量 / 它的低秩代理 | `HH` / `Z` |
| $Y_n$ | $n\times l$ | **要学**：显式噪声标签 | `Y_n`，随机二值 |

**"全局权重 → 逐视图权重"是怎么做的（论文 Figure 2 的技巧）**：TOCL 并不为每个视图单独学一个
$W^{(i)}$（那样视图之间不再共享任何东西，等于跑 $V$ 个独立模型）。它只学**一个**全局
$W\in\mathbb R^{d\times l}$，再用 $A^{(i)}\in\mathbb R^{d^{(i)}\times d}$ 把属于第 $i$ 个视图的行块取出来：
$U^{(i)}=A^{(i)}W\in\mathbb R^{d^{(i)}\times l}$。**共享 $W$ 是"跨视图联合选特征"的全部机制**，
$A^{(i)}$ 只是一个固定的选择器（`alg/TOCL.py` L100–111 用对角线置 1 构造，之后再也不更新）。

**逐项解剖**（配合式 (3.1)–(3.7)）

| 项 | 数学形式 | 在说什么 | 为什么需要它 |
|---|---|---|---|
| ① 局部线性 | $\sum_i\|PX^{(i)}A^{(i)}W-y_t^{(i)}\|_F^2$ | 对每个视图：用**该视图自己的标签** $y_t^{(i)}$ 监督它自己的特征块 | 这是"互补性"的来源 —— 每个视图被允许有自己的预测目标，不必都去拟合同一个 $Y$ |
| ② 全局非线性 | $\alpha\|PXW-FU\|_F^2+\|U\|_F^2+\|V\|_F^2$ | 让**线性分支的预测** $PXW$ 与 **sigmoid 分支的预测** $FU$ 互相一致 | "对立互补"的另一半，见下面的"监督链" |
| ③ 张量低秩 | $\|Z-T\|_F^2+\|T\|_{w,*}$ | 让 $V$ 个视图的标签关联矩阵共享同一套低秩结构；$Z$ 是 $T$ 的低秩代理 | 跨视图**高阶**约束 + 标签去噪 —— 本文最核心的贡献 |
| ④ 离散标签相关 | $\gamma(\|Y_{all}\otimes Y_n\ominus Y\|_F^2+\|Y_n\|_F^2)$ | 观测标签必须能被"某些视图说它相关"（$Y_{all}$）**或**"它是噪声"（$Y_n$）解释 | 把 $y_t$ 与 $Y$ 连起来（否则 $y_t$ 完全自由），同时给噪声一个出口 |
| ⑤ 正则 | $\beta\|P\|_F^2+\delta\|W\|_{2,1}$ | 限制 $P$ 的尺度；让 $W$ **行稀疏** | $\ell_{2,1}$ 行稀疏就是把"特征重要性 = 行 2-范数"这件事直接写进模型 |

**两个容易看漏的细节**

* **$P$ 不是对角阵。** 论文文字称 $P$ 为"实例级重要性矩阵"（instance-level importance），
  直觉上应该是对角权重 $\mathrm{diag}(p)$；但式 (3.1) 与代码（`P = rng.random((num, num))`、
  `np.dot(P, X)`）里的 $P$ 都是**稠密 $n\times n$** —— 它在**样本方向**做线性混合，而不是逐样本加权。
  β 项 $\|P\|_F^2$ 的作用正是防止这个 $n\times n$ 矩阵任意放大。读论文时按"样本混合矩阵"理解更贴合实现。
* **式(14) 与式(10) 的记号不一致**：式(10) 写 $\|T\|_{w,*}$（加权），式(14) 写 $\tau\|T\|_*$（标准）。
  代码走的是**加权版**（`rho=1`，见 §3.1.5）。

**监督链（理解 TOCL 的关键）**：把五项连起来看，信息是单向流动的：

```
Y ──④离散标签相关──► y_t⁽ⁱ⁾ ──①线性拟合──► W ──②两路一致──► U, V
                                          │
                                          └──⑤ℓ2,1──► 特征排序 ‖W_(j)‖₂
```

注意 **$U,V$ 没有直接挂在 $Y$ 上**：非线性分支唯一的监督信号来自"与线性分支一致"（②），
所以它学到的是线性分支预测的一个**平滑（概率化）版本**，而不是直接拟合标签。这正是论文把它称为
"互补"的原因 —— 线性分支给非线性分支当老师，非线性分支再通过 $P$ 反向影响线性分支。

**维度自检**（这类模型读不动时，先确认每个乘积能对上）

$$
\underbrace{P}_{n\times n}\underbrace{X^{(i)}}_{n\times d^{(i)}}\underbrace{A^{(i)}}_{d^{(i)}\times d}\underbrace{W}_{d\times l}=n\times l=\dim y_t^{(i)},
\qquad
\underbrace{F}_{n\times k}\underbrace{U}_{k\times l}=n\times l=\dim(PXW)
$$

$$
T_{:,:,i}=\underbrace{y_t^{(i)}}_{n\times l}\underbrace{y_t^{(i)\top}}_{l\times n}=n\times n
\;\Longrightarrow\; T\in\mathbb{R}^{n\times V\times n}
$$

#### 3.1.3 "对立又互补"的确切含义

论文原文：
"during the learning phase, the **local learning process focuses on extracting distinctive
features**, while the **global learning process mitigates feature redundancy across views**
by assessing feature weights across all views. This dual process allows for a finer-grained
understanding of view importance… Although these processes may appear **contradictory**,
they are **mutually reinforcing**, which is why we characterize our method as
'Opposing yet Complementary'."

即：$P$ 这个视图融合矩阵同时被两个方向牵引 ——

* **局部项** $\sum_i\|PX^{(i)}A^{(i)}W-y_t^{(i)}\|^2$ 让 $P$ 对**各视图的差异**敏感（"对立"）；
* **全局项** $\|PXW-FU\|^2$ 让 $P$ 追求**全局最优**并保证迭代稳定（"互补"）。

$$
P\leftarrow P\circ\frac{\delta\sum_i y_t^{(i)}W^\top A^{(i)\top}X^{(i)\top}+\alpha F^\top U W^\top X^\top}
{\delta\sum_i PX^{(i)}A^{(i)}WW^\top A^{(i)\top}X^{(i)\top}+\alpha BWW^\top X^\top+\beta P}
\tag{3.8}
$$

分子是两个方向的和，分母也是 —— 这就是"对立互补"在更新式里的**代数痕迹**
（乘性更新的通用推导见 §1.5.3：分子是负梯度部分、分母是正梯度部分）。

#### 3.1.4 创新点归因：为什么是 ACM MM (CCF-A)

| 类别 | 创新点 | 证据 | 为什么够 A |
|---|---|---|---|
| **机制（主）** | **把视图特有标签的高阶关系升到张量**：$T\in\mathbb{R}^{n\times V\times n}$ + **加权**张量核范数。首次在多视图多标签特征选择中引入 TNN | 式(10)(22)-(25)；Def.2 | **J1：引入新的数学工具（t-SVD/TNN）与本领域问题结合** |
| **机制（主）** | **加权** TNN：逐奇异值权重 $w_i=C/(\sigma_i+\varepsilon)$，使大奇异值（跨视图主成分）少惩罚、小奇异值（噪声）多惩罚 | Eq.24 前的论证："the fixed threshold τ in this method **fails to recognize that larger singular values, which capture the principal directions of the data**, should be penalized less" | 这是对已有 TNN 的**实质性改进**，且论证清晰 |
| **机制** | **全局非线性 + 局部线性双路映射**，并用 $P$ 把两者耦合（"对立互补"） | 式(9)(16)(26)；Figure 1/2 | 新叙述框架 |
| **建模** | **分层标签相关**（Definition 3 第一层 $Y=Y_{all}\otimes Y_n$；Definition 4 第二层 $Y_{all}=\bigotimes_i y_t^{(i)}$） | Def.3/Def.4；式(4)(5) | 沿用 DHLI 设定但补上了"两层"的形式定义 |
| **建模** | 离散分歧算子 $\ominus$ + 稀疏噪声 $Y_n$ | 式(12)(13) | 与 DHLI 同源 |
| **实验** | Friedman 检验（AP 6.629 / Coverage 9.100 / HL 9.730 / RL 6.248，均 > 临界值 2.244） | Table 4 | — |

**TOCL 是六篇里"新工具引入"最彻底的一篇**，这也是它能上 ACM MM（多媒体领域 A 会，
对"张量/多模态表示"这类工具性创新接受度高）的原因。

**实验数字（AP ↑）**：SCENE 0.7981、OBJECT 0.4959、MIRFlickr 0.6908、Corel5K 0.2419、
IAPRTC12 0.1510、3Sources 0.4883。
TOCL 在 MIRFlickr 上的领先幅度（0.6908 vs MDFS 0.6848）与 3Sources（0.4883 vs EF²FS 0.4608）
相对明显，是其最强的两处证据。

**消融（Table 5, AP）**：四个组件（GN 全局非线性 / DLC 离散标签相关 / TC 张量约束 / LL 局部线性）
逐一累加：M1 0.7792 → M2 0.7850 → M3 0.7850 → M4 0.7949 → TOCL 0.7959（SCENE）。
**注意 M2 与 M3 数值完全相同**（0.7850），论文未解释。

#### 3.1.5 代码对照：公式 ↔ `alg/TOCL.py`

| 论文公式 | 代码位置 | 对应关系 |
|---|---|---|
| 式(16) $W$ 更新 | L189–192 | 分子 `delta*tem1 + alpha*XᵀPᵀ·NL·U`；分母 `delta*tem2 + lamb*D2W + alpha*XᵀPᵀPXW` |
| 式(17) $P$ 更新 | L195–202 | 分子 = 局部（`delta*tem1`）+ 全局（`alpha*NL·U·WᵀXᵀ`）；分母含 `beta*P` |
| 式(18) $Y_n$ 更新 | L205–209 | `tem3 = 1[Y − Y_all ≠ 0]`，再除以 `2*Y_n` |
| 式(19)(20) $U,V$ 更新 | L213、L216–218 | `NL = 1/(I+exp(−XV))`；`Q = NL∘(1−NL)` 正是 sigmoid 的导数 |
| 式(21) $y_t^{(i)}$ 更新 | L156–157 | 分子含 `Z[:,:,i]·y[i]`（张量项的贡献）与 `gamma*F` |
| 式(10)(22)–(25) 加权 TNN | L17–68 | `prox_weight_tensor_nuclear_norm`（见下） |
| 式(14) 目标值 | L225–248 | `temp1..temp9`；注意 `2*beta*temp6`、`2*lamb*temp9` 多出因子 2（§5.2 L 行） |

**① 张量构造 + 加权 TNN 近端算子**（TOCL 的心脏，L167–179）

```python
# L167    T 的第 i 个前切片 = 视图 i 的标签 Gram 矩阵（即 Eq.(10) 的 T_{:,:,i}）
HH[:, :, i] = np.dot(y[i], y[i].T)

# L177-179 转置后送入加权 TNN 近端算子（rho = C），再转回来
HH2 = HH.transpose((0, 2, 1))
Z2, wtnn, trank = prox_weight_tensor_nuclear_norm(HH2, rho)
Z = Z2.transpose((0, 2, 1))
```

```python
# L17-68  加权 TNN 近端算子 = FFT → 逐切片 SVD → 闭式收缩（式24）→ iFFT
def prox_weight_tensor_nuclear_norm(Y, C):
    n1, n2, n3 = Y.shape
    X = np.zeros((n1, n2, n3), dtype=complex)
    Y = fft(Y, axis=2)                      # 式(3) 的"沿第 3 维做 DFT"
    ...
    temp = (S - eps) ** 2 - 4 * (C - eps * S)                  # 式(24) 的判别式 c2
    ind = np.where(temp > 0)[0]                                # c2 < 0 的奇异值直接置 0
    S = np.maximum(S[ind] - eps + np.sqrt(temp[ind]), 0) / 2   # 式(24) 的 (c1+√c2)/2
    X[:, :, 1] = U[:, :r] @ np.diag(S) @ V[:, :r].T            # 式(25) 重构切片
    wtnn += np.sum(S * (C / (S + eps)))                        # w_i = C/(σ_i+ε)，对应式(22)
    ...
    newX = np.fft.ifftn(X)                  # 回到空域
    wtnn /= n3                              # 式(22) 的 1/n 归一化
    return np.real(newX), wtnn, trank
```

读这一小段请务必对照 §5.2 的 TOCL 行：`range(1, halfn3)` **从不处理第 0 个切片**（直流分量被丢）、
`X[:, :, n3-i] = conj(X[:, :, i])` 会**覆盖**后面已经算出的切片、偶数视图时 `i = halfn3 + 1` **越界**
（$V=2$ 直接崩）。`insight/mechanisms.py::prox_weighted_tnn` 是修好这些问题的版本，
`python -m insight.run_demo M1` 会把它与原实现**逐值对照**。

**② 目标函数**（L225–248）

```python
temp1 = Σ_i ‖P·x[i]·d[i]·W − y[i]‖_F²           # ① 局部线性（式 3.1）
temp3 = 0.5*temp2 + wtnn                         # ③ ‖Z−T‖²_F 的一半 + 加权 TNN 值（式 3.3）
temp4 = ‖PXW − NL·U‖_F²                          # ② 全局非线性（式 3.2）
temp5 = ‖U‖_F² + ‖V‖_F²
temp7 = ‖1[Y_all + Y_n − Y ≠ 0]‖_F²,  temp8 = ‖Y_n‖_F²    # ④ 离散标签相关（式 3.6）
objectives = delta*temp1 + aaa*temp3 + alpha*(temp4+temp5) \
             + 2*beta*temp6 + gamma*(temp7+temp8) + 2*lamb*temp9
```

**③ 三处"公式 vs 代码"的差异（读代码时必须知道）**

| 现象 | 说明 | 影响 |
|---|---|---|
| `y[i]` 每轮被 MinMax + 0.5 **硬二值化**（L160–165） | 论文只在 Definition 4 声明 $y_t^{(i)}\in\{0,1\}$，正文的乘性更新按**连续变量**推导 | 与 §5.2 中 DHLI / I²VSLC 的 H 级偏差同类：乘性更新之间插入硬投影，破坏"目标单调"的前提 |
| 目标值里 `2*beta`、`2*lamb` 的因子 2 | 论文式(14) 写的是 $\beta\|P\|_F^2$ 与 $\delta\|W\|_{2,1}$ | 只影响停机判据（§5.2 L 行） |
| `Y_n` 的**更新式**与**目标值判据**不同步 | 更新用 `Y − Y_all`，目标值却用 `Y_all + Y_n − Y` 的非零位置；在 $Y_{all}=Y_n=Y=1$ 处目标值算它"违规"而更新式不管它 | M 级：$\|Y_n\|_F^2$ 本身仍在惩罚 $Y_n=1$，实际影响小，但说明**记录的目标值并不是驱动迭代的那个函数**（同 §5.7 的批评） |

---

### 3.2 UGRFS — 不确定度感知的全局视图重构（AAAI 2025）

> **一句话**：把"每个视图、每个样本的监督都同等可信（可信度 ≡ 1）"这条**公理放宽成可学变量** ——
> 用 $\mathrm{diag}(C^{(i)})$（样本级置信度）同时挂在回归损失和"全局视图拆分"罚项两端；
> 同时**不显式切分**一致/互补，而是学一个全局视图变量 $D$，让模型自己决定怎么分。

**这篇的抓手：把"可信度"从公理变成变量**

* 前几篇都默认 $X^{(i)}W^{(i)}\approx Y$ 里每个样本、每个视图同等重要。UGRFS 反问：一张病理图像可能
  拍糊了、一行表格数据可能整行是噪声，**凭什么给它同样的权重**？于是引入 $C^{(i)}\in\mathbb R^n$，
  用 $\mathrm{diag}(C^{(i)})$ 做**逐样本缩放**（注意不是 §3.1 里那种 $n\times n$ 的混合矩阵 $P$）。
* 第二个抓手是 D1 的解法：**不切分**。"一致/互补"的切分点历来是人工设定的，UGRFS 直接学一个
  全局视图 $D\in\mathbb R^{n\times d}$，把它同时当"融合结果"和"重构目标"。
* 读这篇时先把**两个 $D$** 分清：论文的 $D$ 是**全局视图分布**（$n\times d$），代码里是 `p2 = Yx@Wy`；
  而论文的 $E$（$\ell_{2,1}$ 的重加权对角阵）在代码里叫 `D`。极易误读，§5.2 有记录。

#### 3.2.1 它认为前人错在哪（四类缺陷）

| 缺陷 | 原文 | 中译 |
|---|---|---|
| **D1 一致/互补分开提取 ⇒ 分割不清** | "Existing methods often extract information **separately** from the consistency part and the complementary part, which may result in noise due to **unclear segmentation**" | 分开提取导致切分不清引入噪声 |
| **D2 直接拼接忽略两种关系** | "they often overlook the **view relationship** and fail to consider the **sample-level relationship**… there most likely exists noise and redundancy in direct concatenating matrix" | 忽略视图关系与样本级关系 |
| **D3 样本可信度被默认 = 1** | "it is worth considering the **relaxation of the assumption that the reliability of each view, and even each sample, is uniformly equal to 1**" | 这是本文的核心动机 |
| **D4 多对一映射** | "the effective learning of weights for all features remains a challenging task due to the inherent **many-to-one mapping** relationship" | 多对一使权重难学 |

**D1 的解法非常值得注意**：论文不是"更好地分割"，而是**不分割** ——
"the new variable for global-view is automatically learned instead of being fused with
**predefined weights**. This allows for balancing the consistency and complementary
**within one variable**."

**D3 的现实依据**（这条论证质量很高）：
"In reality, the quality of **histopathological images** may vary across different patients,
and **tabular data** may contain feature noise. As a result, there can be disparate
confidence levels among samples."

论文用 Figure 1 归纳了四种"特征–标签关系"范式（a 只取共识 / b 标签一致映射 /
c 视图特有映射 / d 拼接），逐一指出问题，把自己的定位放在这四者之外。

#### 3.2.2 目标函数

$$
\begin{aligned}
\min_{W^{(i)},C^{(i)},W_y^{(i)}}\;
&\sum_{i=1}^{V}\big\|\mathrm{diag}(C^{(i)})X^{(i)}W^{(i)}-Y\big\|_F^2
+\alpha\,\mathrm{Tr}\big(D^\top L_YD\big)\\
&+\beta\sum_{i=1}^{V}\big\|D^{(i)}-\mathrm{diag}(C^{(i)})X^{(i)}\big\|_F^2
+\gamma\big\|D-X^f\big\|_F^2
+\delta\|W\|_{2,1}
\end{aligned}
\tag{3.9}
$$

其中 $D=Y_xW_y$，$Y_x=[\rho(Y),J_1]\in\mathbb{R}^{n\times(n+1)}$，$X^f=[v_1X^{(1)},\dots,v_VX^{(V)}]$。

> **⚠️ 论文 Eq.(7) 的印刷问题**：$D=\rho(Y)\hat W_y+b$ 与
> $\hat W_y=[W_y;b]\in\mathbb{R}^{(n+1)\times d}$、$\rho(Y)\in\mathbb{R}^{n\times n}$
> **量纲不自洽**（$n\times n$ 无法乘 $(n+1)\times d$）。唯一自洽的矩阵形式是
> $D=Y_xW_y$，代码实现的正是后者。

**$\rho(Y)$ 的高斯核映射**（式 6）：

$$
\rho(Y)=\exp\!\left(-\frac{J_1Y^\top\big(\sum_{i=1}^{l}(y_{ij})^2\big)-YY^\top}{\big(\mathrm{avg}(p\,dist)\big)^2}\right)
\tag{3.10}
$$

**乘性更新**（式 14–16）：

$$
\begin{aligned}
W^{(i)} &\leftarrow W^{(i)}\circ\frac{A^{(i)\top}Y}{A^{(i)\top}A^{(i)}W^{(i)}+2\delta EW^{(i)}}\\[2pt]
C^{(i)} &\leftarrow C^{(i)}\circ\frac{YW^{(i)\top}X^{(i)\top}+\beta D^{(i)}X^{(i)\top}}{A^{(i)}W^{(i)}W^{(i)\top}X^{(i)\top}+\beta A^{(i)}X^{(i)\top}}\\[2pt]
W_y^{(i)} &\leftarrow W_y^{(i)}\circ\frac{\alpha Y_x^\top S_YD^{(i)}+\beta Y_x^\top A^{(i)}+\gamma Y_x^\top X^{f(i)}}{\alpha Y_x^\top A_YD^{(i)}+(\beta+\gamma)Y_x^\top D^{(i)}}
\end{aligned}
\tag{3.11}
$$

其中 $A^{(i)}=\mathrm{diag}(C^{(i)})X^{(i)}$，$E_{pp}=1/(2\|W_p\|_2)$。

**变量清单：谁是已知量，谁是要学的量**

| 符号 | 维度 | 角色 | 代码里的名字 |
|---|---|---|---|
| $X^{(i)}$（拼接为 $X$） | $n\times d^{(i)}$（$n\times d$） | 已知特征 | `x[i]` / `X` |
| $Y$ | $n\times l$ | 已知标签 | `Y` |
| $W^{(i)}$ | $d^{(i)}\times l$ | **要学**：每个视图自己的特征权重 | `w[i]` |
| $W=\bigoplus_i W^{(i)}$ | $d\times l$ | 排序依据：**纵向拼接**后的整体权重 | `W = np.concatenate(w, axis=0)` |
| $C^{(i)}$ | $n$（对角） | **要学**：样本级置信度 | `c[i] = np.diag(...)` |
| $D$（全局视图分布） | $n\times d$ | **要学**：学出来的全局视图 | `p2 = Yx @ Wy` |
| $D^{(i)}$ | $n\times d^{(i)}$ | **要学**：$D$ 在第 $i$ 个视图上的分量 | `Yx @ wy[i]` |
| $W_y^{(i)}$ | $(n{+}1)\times d^{(i)}$ | **要学**：从标签核空间回归出全局视图的系数 | `wy[i]` |
| $v_i$ | 标量 | **算出来**：视图权重（不是自由变量） | `nu[i]` |

**逐项解剖**（式 3.9）

| 项 | 在说什么 | 为什么需要 |
|---|---|---|
| $\sum_i\|\mathrm{diag}(C^{(i)})X^{(i)}W^{(i)}-Y\|_F^2$ | 每个视图单独回归标签，但**每个样本的误差按可信度缩放** | 置信度低 ⇒ 该样本的误差被自动压小 ⇒ 不污染 $W^{(i)}$ |
| $\alpha\,\mathrm{Tr}(D^\top L_YD)$ | 全局视图 $D$ 在**标签图** $L_Y$ 上要平滑：标签相近的样本，其全局视图表示也应相近 | $D$ 的结构先验。注意 $L_Y$ 建在**标签**上（代码 L66–71），与 I²VSLC 把图建在特征上正好相反 |
| $\beta\sum_i\|D^{(i)}-\mathrm{diag}(C^{(i)})X^{(i)}\|_F^2$ | 全局视图要能被**拆回**各视图的（置信度加权）特征 | 这就是"一致/互补不切分"的落地：$D^{(i)}$ 即第 $i$ 个视图贡献的那一块 |
| $\gamma\|D-X^f\|_F^2$ | 全局视图还要接近"老式"的视图加权拼接 $X^f=[v_1X^{(1)},\dots,v_VX^{(V)}]$ | 拿拼接当锚点，防止 $D$ 漂移 |
| $\delta\|W\|_{2,1}$ | 行稀疏 | 排序依据 |

**变量之间的闭环**（这张图解释了"不确定度感知的重构"到底在循环什么）

```
            C⁽ⁱ⁾（样本可信度）
            ▲  ↖________ 由 W⁽ⁱ⁾ 的残差推动
            │
   diag(C⁽ⁱ⁾)X⁽ⁱ⁾ ──β──► D⁽ⁱ⁾ ◄── 拆出 ── D ──γ──► X^f=[v₁X⁽¹⁾,…]
        ▲                                    ▲
        └───────── W⁽ⁱ⁾ 拟合 Y ──────────────┘    v_i = 1/Tr(X⁽ⁱ⁾ᵀL_yX⁽ⁱ⁾)（代码：只算一次）
```

$C^{(i)}$ 出现在**两处**（回归损失 + 拆分罚项），这是论文自称"置信度与特征权重互为条件、
互相促进"的代数依据。

**维度自检**

$$
\mathrm{diag}(C^{(i)})X^{(i)}W^{(i)}=(n\times n)(n\times d^{(i)})(d^{(i)}\times l)=n\times l=\dim Y
$$
$$
Y_xW_y^{(i)}=\underbrace{(n\times(n+1))}_{[\rho(Y),J_1]}\underbrace{((n+1)\times d^{(i)})}_{W_y^{(i)}}=n\times d^{(i)}=\dim D^{(i)}
$$

**更新式从哪来**：与 §1.5.3 的通用推导完全同一套（非负 + KKT ⇒ 分子是负梯度部分、分母是正梯度部分），
这里只补两个实现细节：$E_{pp}=1/(2\|W_p\|_2)$ 是把 $\|W\|_{2,1}=\sum_p\|W_p\|_2$ 用辅助函数
$\|W_p\|_2\approx W_p^\top EW_p$ 松弛后得到的（六篇同构）；$Y_x=[\rho(Y),J_1]$ 里的 $J_1$ 是全 1 列，
相当于给线性回归补了一个偏置列。

#### 3.2.3 创新点归因：为什么是 AAAI (CCF-A)

| 类别 | 创新点 | 证据 | 为什么够 A |
|---|---|---|---|
| **机制（主）** | **样本级不确定度作为可学习变量进入重构机制**：$\mathrm{diag}(C^{(i)})$ 同时挂在"视图→标签"损失与"全局视图拆分"罚项两端，使置信度与特征权重**互为条件、互相促进** | 贡献 1；式(9)(10)(11)；Algorithm 1 第 4 步 | **J1：引入新数学对象 $C^{(i)}$**。前人把"样本可信度恒为 1"当公理，UGRFS 把它当变量 |
| **机制（主）** | **用一个"全局视图"变量取代显式的一致/互补分解** —— 不是更好地分割，而是不分割 | 贡献 2；式(5)(8)(12) | **J2：把人工先验（如何分割）变成可学变量** |
| **机制** | **双向结构对齐**：$D\to D^{(i)}$ 下传到各视图（$\beta$），同时各视图 $\to Y$（主项）；全局侧另有 $D\to X^f$（$\gamma$）与 $D$ 在 $L_Y$ 上平滑（$\alpha$） | 式(12) 的四项 | 四面牵引的联合模型 |
| **机制** | 标签矩阵的**非线性核映射作为回归目标空间**（把 $Y$ 提升到 $n\times n$ 再回归出 $D$） | 式(6)(7) | — |
| **实验** | 6 数据集 × 6 基线 × 4 指标 + **三变体消融** + **样本置信度可视化**（Fig.4a/b） | Table 2–4；Fig.4 | 可视化证据是它的加分项 |

**为什么够 A 而 GRAFS 只够 B（两者结构其实很像）**：

这是本文档最值得注意的一组对比。UGRFS 与 GRAFS 的骨架高度相似（都是"学一个全局视图
$D$/$X^f$" + "视图权重 $v_i$" + "全局视图拆回各视图"），但：

| 维度 | GRAFS (B) | UGRFS (A) | 差别 |
|---|---|---|---|
| 新数学对象 | $X^f$（全局视图） | $X^f$ **+ $C^{(i)}$（样本置信度）** | **UGRFS 多一个 J1 级对象** |
| 动机的现实锚点 | 复杂度（工程性） | "病理图像质量因患者而异"（**领域性**） | UGRFS 的动机更难被质疑 |
| 结构参数敏感性 | $k,k_1$ **完全未分析** | $\alpha,\beta,\gamma,\delta$ 全扫；无额外结构参数 | UGRFS 更完整 |
| 可视化证据 | 收敛曲线 | 收敛曲线 **+ 置信度热图 + 特征图结构变化对比** | UGRFS 多一类证据 |
| 消融设计 | 2 变体 | 3 变体（**v1 专测样本置信度**） | 直击自己的核心主张 |

**结论**：J1（是否有新数学对象）+ 动机的领域锚点是 A/B 分界的实际判据。

**实验数字（AP ↑）**：yeast 0.6725、SCENE 0.8010、VOC07 0.5871、MIRFlickr 0.6770、
IAPRTC12 0.1474、3Sources 0.4728。
**优势幅度**：论文称"比第二名高 **4.84%**、比最后一名高 14.3%"，但按其表内数字逐数据集
复核，相对当轮次优的相对提升是 MIRFlickr 5.09%、IAPRTC12 4.10%、3Sources 2.27%、
yeast 1.97%、SCENE 1.02%、VOC07 0.60%（均值 ≈2.5%）。**4.84% 的聚合口径论文未说明。**

**消融（Table 4）**：UGRFSv1（去样本置信度）MIRFlickr AP 0.6770→0.6646（$-1.8\%$）；
v3（去全部重构正则，$\alpha=\beta=\gamma=0$）0.6770→0.6470（$-4.4\%$）。
**注意 v2（忽略视图加权）有时比 v3 还差**，论文据此论证"正确计算视图权重"的重要性 ——
而 §5 会指出这个视图权重公式本身有缺陷。

#### 3.2.4 代码对照：公式 ↔ `alg/UGRFS.py`

| 论文公式 | 代码位置 | 对应关系 |
|---|---|---|
| 视图权重 $v_i\propto1/\mathrm{Tr}(y^\top L_x^{(i)}y)$ | L74–79 | 代码算的是 $1/\mathrm{Tr}(X^{(i)\top}L_yX^{(i)})$ —— **角色互换**（§5.2 H 行）；且它在 `while` **循环之外**只算一次 |
| 标签图 $L_Y$ | L64–71 | `construct_W(Y, knn k=20, heat_kernel, t=1.0)` → `Ly = Ay − Sy` |
| $\rho(Y)$ 核映射 | L86–89 | `par = 1/mean(pdist(Y))`；`H = exp(−D/(2 par²))`；`Yx = [H, 1]` |
| 式(14) $W^{(i)}$ | L106–112 | `D = diag(0.5/‖w_i‖)`；分子 `p1ᵀY`，分母 `p1ᵀp1·w + lamb·D·w` |
| 式(15) $C^{(i)}$ | L110–117 | 乘性更新后 `np.diagonal(...)` 再 `np.diag(...)` ⇒ **强制成对角阵** |
| 式(16) $W_y^{(i)}$ | L118–120 | 分子 `gamma·Yxᵀp3 + alpha·YxᵀSyYx·wy + beta·Yxᵀc[i]x[i]` |
| 式(12) 目标值 | L124–138 | `temp1..temp5`；`temp5 = Σ_p‖W_p‖_2` |

```python
# L104-120  一轮里三个变量依次乘性更新（分子=负梯度部分，分母=正梯度部分）
for i in range(n_view):
    d1 = 0.5 / np.sqrt(np.sum(w[i] * w[i], 1) + eps)   # E_pp = 1/(2‖W_p‖₂) 的那个 0.5
    D = np.diag(d1.flat)
    p1 = np.dot(c[i], x[i])                            # A⁽ⁱ⁾ = diag(C⁽ⁱ⁾)X⁽ⁱ⁾
    w[i] = np.multiply(w[i], np.true_divide(
        np.dot(p1.T, Y),
        np.dot(np.dot(p1.T, p1), w[i]) + lamb * np.dot(D, w[i]) + eps))

    c[i] = np.multiply(c[i], np.true_divide(
        beta * np.dot(np.dot(Yx, wy[i]), x[i].T) + np.dot(np.dot(Y, w[i].T), x[i].T),
        beta * np.dot(np.dot(c[i], x[i]), x[i].T)
        + np.dot(np.dot(np.dot(p1, w[i]), w[i].T), x[i].T) + eps))
    c[i] = np.diagonal(c[i]).reshape(num, 1)           # ← 只取对角：C⁽ⁱ⁾ 被强制成对角阵
    c[i] = np.diag(c[i].flat)

    p3 = x[i] * nu[i]                                  # v_i·X⁽ⁱ⁾，即 X^f 的第 i 块
    wy[i] = np.multiply(wy[i], np.true_divide(
        gamma * np.dot(Yx.T, p3) + alpha * np.dot(np.dot(np.dot(Yx.T, Sy), Yx), wy[i])
        + beta * np.dot(np.dot(Yx.T, c[i]), x[i]),
        alpha * np.dot(np.dot(np.dot(Yx.T, Ay), Yx), wy[i])
        + (gamma + beta) * np.dot(np.dot(Yx.T, Yx), wy[i]) + eps))
```

**读这段代码要带上 §5.2 的 UGRFS 六条**：$W^{(i)}$ 的 $\ell_{2,1}$ 有效权重只有论文的一半
（`0.5/·` 对论文的 $2\delta$，要与论文一致需把 `lamb` 取成 $2\delta$）；$C^{(i)}$ 的分母用的是
**旧的** `p1`，却配刚更新的 `w[i]`（Jacobi 式混合，论文未声明）；$\rho(Y)$ 被换成一个带宽方向
相反的普通高斯核；视图权重公式的角色被互换。另外 **L138 的 `objectives` 与 L141 的 `objectives/2`
正好解释了论文 Fig.5 里"首轮相对变化率 ≈ 1"**（§5.2 L 行）。

---

### 3.3 EF²FS — 嵌入式特征融合（Pattern Recognition 2025）

> **一句话**：把"**先算视图权重 → 拼出 $X^f$ → 再选特征**"这条两步流水线**并进一个损失函数**：
> 视图权重 $c_i$ 由隐标签 $G$ 的图能量决定，$G$ 又由融合特征 $X^fA$ 和各视图 $x^{(i)}w^{(i)}$ 共同拟合 ——
> 融合与选择互相指导。

**这篇的抓手：融合不再是预处理**

前人（含 GRAFS 引用的 Zhu et al.）的流程是**两步**：先按某个准则把多视图加权拼成 $X^f$，
再在 $X^f$ 上跑 $\ell_{2,1}$ 回归选特征。EF²FS 认为两步之间信息断裂（第一步并不知道第二步要什么），
于是把融合权重写进损失：$c_i$ 是 $G$ 的函数，而 $G$ 又被所有视图与标签同时拟合。

#### 3.3.1 它认为前人错在哪

| # | 原文 | 中译 | 指向 |
|---|---|---|---|
| S1 | "Existing methods often extract information **separately** from the consensus part and the complementary part, potentially leading to noise attributed to **ambiguous segmentation**" | 分别提取一致/互补 ⇒ 切分边界模糊引入噪声 | 分治式流水线 |
| S2 | "It is imperative to define and constrain the **segment** among views to prevent overlaps or omissions" | 必须界定视图划分以防重叠/遗漏 | 切分点人为设定 |
| S3 | "Zhu et al. propose a **two-step** method… the existing methods for obtaining the comprehensive part are relatively limited, often relying on simple techniques such as **direct concatenation**" | 两步法 + 拼接手段贫乏 | 明确点名"两步" |
| S4 | "these methods often **overlook the design of a regularizer to preserve distinctive information**" | 视图加权类工作缺"保独有信息"的正则 | 批 [36][37] |
| S5 | "the utilization of label formation encounters inherent limitations within the **unsupervised** domain" | 无监督自适应视图权重**用不上标签** | 批 [9][10][11] |

#### 3.3.2 目标函数

$$
\min_{A,G,B,w^{(i)}}\;
\underbrace{\big\|X^fA-G\big\|_F^2}_{\text{融合→嵌入}}
+\alpha\underbrace{\big\|Y-GB^\top\big\|_F^2}_{\text{嵌入→标签}}
+\beta\underbrace{\sum_i\big\|x^{(i)}w^{(i)}-G\big\|_F^2}_{\text{各视图→嵌入}}
+\gamma\underbrace{\sum_i\big\|a^{(i)}-w^{(i)}\big\|_F^2}_{\text{全局块↔局部权重}}
+\delta\underbrace{\|A\|_{2,1}}_{\text{行稀疏}}
\tag{3.12}
$$

**视图权重**（论文最终采用 $G$ 版，式 12）：

$$
c_i=\frac{1/\mathrm{Tr}\big(G^\top L^{(i)}G\big)}{\sum_{j=1}^{V}1/\mathrm{Tr}\big(G^\top L^{(j)}G\big)},
\qquad X^f=[c_1x^{(1)},\dots,c_Vx^{(V)}]
\tag{3.13}
$$

**推导补充**（论文只给结果并引 [37]，这里补上闭式来源）：$c_i\propto 1/E_i$ 正是

$$
\min_{c_i\ge0,\ \sum_i c_i=1}\ \sum_{i=1}^{V}c_i^2E_i
\quad\Longrightarrow\quad 2c_iE_i=\lambda\ \Rightarrow\ c_i\propto\frac{1}{E_i}
\tag{3.14}
$$

的闭式解。**权重取平方形式**（而非线性 $\sum_i c_iE_i$，后者会退化成"全部权重压到
能量最小的那个视图"的角点解）——这一点的动机在论文中没有写清楚，但对理解
§5 的缺陷至关重要。

**乘性更新**（式 21/23/25/27）：

$$
\begin{aligned}
G &\leftarrow G\circ\frac{\alpha YB+\beta\sum_i x^{(i)}w^{(i)}+X^fA}{\alpha GB^\top B+(\beta V+1)G},&
B&\leftarrow B\circ\frac{Y^\top G}{BG^\top G}\\
A &\leftarrow A\circ\frac{X^{f\top}G+\gamma W}{X^{f\top}X^fA+\gamma A+\delta DA},&
w^{(i)}&\leftarrow w^{(i)}\circ\frac{\beta x^{(i)\top}G+\gamma a^{(i)}}{\beta x^{(i)\top}x^{(i)}w^{(i)}+\gamma w^{(i)}}
\end{aligned}
\tag{3.15}
$$

**变量清单：谁是已知量，谁是要学的量**

| 符号 | 维度 | 角色 | 代码里的名字 |
|---|---|---|---|
| $x^{(i)}$ | $n\times d^{(i)}$ | 已知特征块 | `v[i]` |
| $Y$ | $n\times l$ | 已知标签 | `Y` |
| $X^f=[c_1x^{(1)},\dots,c_Vx^{(V)}]$ | $n\times d$ | **算出来**：视图加权拼接 | `new_X` |
| $A$ | $d\times k$ | **要学**：融合后的特征权重 = **最终排序依据** | `A` |
| $G$ | $n\times k$ | **要学**：隐标签 / 公共嵌入（$k$ = 隐空间维数） | `V` ⚠️ 与"视图数"无关，极易误读 |
| $B$ | $l\times k$ | **要学**：从嵌入 $G$ 回到标签的映射 | `B` |
| $w^{(i)}$ | $d^{(i)}\times k$ | **要学**：每个视图各自的局部投影 | `w[i]` |
| $c_i$ | 标量 | **算出来**：视图权重 | `nu[i]` |
| $L^{(i)}$ | $n\times n$ | 已知：视图 $i$ 的 knn 图拉普拉斯 | `Lx_lst[i]` |

**逐项解剖**（式 3.12）

| 项 | 在说什么 | 为什么需要 |
|---|---|---|
| ① $\|X^fA-G\|_F^2$ | 融合后的特征要能**重构出隐标签** $G$ | 主链：排序依据 $A$ 直接挂在这里 |
| ② $\alpha\|Y-GB^\top\|_F^2$ | $G$ 再线性回归回观测标签 | $G$ 是标签的低维（去噪）版本，用 $B$ 映回 $l$ 维 |
| ③ $\beta\sum_i\|x^{(i)}w^{(i)}-G\|_F^2$ | 每个视图自己的投影也要拟合同**一个** $G$ | 让 $G$ 真正"公共"；各视图的互补信息保留在 $w^{(i)}$ 里 |
| ④ $\gamma\sum_i\|a^{(i)}-w^{(i)}\|_F^2$ | $A$ 中属于视图 $i$ 的行块 $a^{(i)}$ 要与该视图自己的 $w^{(i)}$ **一致** | 迫使"全局融合权重"与"局部视图权重"说同一件事 —— 这是最实质的一条 |
| ⑤ $\delta\|A\|_{2,1}$ | 行稀疏 | 排序 |

**$c_i$ 的闭式来源**（式 3.13/3.14 的补充说明）：$E_i=\mathrm{Tr}(G^\top L^{(i)}G)$ 度量的是
"$G$ 在第 $i$ 个视图的图上有多粗糙"。解 $\min_{c\ge0,\sum c=1}\sum_i c_i^2E_i$ 得 $c_i\propto1/E_i$，
即**哪个视图的图结构与 $G$ 最一致，它的话语权越大**。目标是 $c_i^2E_i$ 而不是线性的 $c_iE_i$ ——
线性形式的最优解会把全部权重压给 $E_i$ 最小的那个视图（角点解），平方形式才有内点解。

**维度自检**

$$
\underbrace{X^f}_{n\times d}\underbrace{A}_{d\times k}=n\times k=\dim G,\qquad
\underbrace{G}_{n\times k}\underbrace{B^\top}_{k\times l}=n\times l=\dim Y,\qquad
a^{(i)}=A\ \text{的第 } i \text{ 个行块}=d^{(i)}\times k=\dim w^{(i)}
$$

**更新式从哪来**：仍是非负 + KKT（见 §1.5.3）。EF²FS 是六篇里**唯一给了单调性论证**的：
论文用 [44] 的辅助函数框架证明了目标单调不增（式 28）—— 注意是"辅助函数式论证"，
不是完整定理（无收敛率、无全局最优性）。

#### 3.3.3 创新点归因：为什么是 PR（CCF-B / 中科院一区 Top）

| 类别 | 创新点 | 证据 | 诚实评价 |
|---|---|---|---|
| **机制（主）** | **融合权重内生闭环**：$c_i$ 是隐标签 $G$ 的图能量闭式函数（式 12），$G$ 又在式(21) 与 $A,B,w^{(i)}$ 联合更新，$X^f$ 反向进入 $G$/$A$ 的更新式 ⇒ 融合↔选择**互相指导** | 论文 §3.2.1；式(9)(12)(21)(25) | $c_i$ 的 $1/\mathrm{Tr}$ 形式**直接借用 [37]**，增量只在"$y\to G$"的耦合 |
| **机制** | **全局-局部双层权重一致性** $\sum_i\|a^{(i)}-w^{(i)}\|_F^2$：一次覆盖 inter-view($A$) 与 intra-view($w^{(i)}$) | 式(14)(15)；Table 8 | 这是最实质的一条 |
| **机制** | 隐标签 $G$ **一物两用**：既是标签去噪器（式 8 的标签簇分解），又是公共嵌入（式 13/14） | 论文 §3.2.2 | — |
| **优化** | 借 [44] 辅助函数框架**证明目标单调不增**（式 28） | 论文 §3.3 末 | **六篇中唯一给出单调性论证的**（虽然只是辅助函数式论证，非完整定理） |
| **实验** | 24 个单元格全部第一 + Friedman 检验（AP 4.368 / Coverage 3.435 / HL 4.912 / RL 4.000，均 > 临界值 2.53） | Table 3–5 | — |

**为什么只到 B**：
1. **论文自认融合项无法删除式消融**："the function of feature fusion operates on the loss function… its functionality **cannot be verified by simply removing relevant elements**"。改用"主/辅成分互换"(ver2) 作**替代指标**，这只证明"谁是主成分"，**没有**证明"在目标函数内部融合"优于"先融合再选择"；
2. **全文没有二阶段流水线的直接基线**（例如"用式(12) 先算 $c_i$ 得到 $X^f$，再跑普通 $\ell_{2,1}$ 回归"）—— 这正是它最核心主张的**空缺对照**；
3. **没有对视图权重机制本身的消融**（$c_i\equiv1$ / 均匀 / 随机 vs 自适应）；
4. Table 7 显示在 IAPRTC12 上 ver2 **反超 3 项**（AP 0.1473 vs 0.1418，+3.88%）；
5. 隐空间维数 $k$ 全文未给取值或选取规则，而代码必须传 `V_dim`（默认 30）。

**实验数字（AP ↑）**：emotions 0.6911、yeast 0.6639、SCENE 0.7935、MIRFlickr 0.6759、
IAPRTC12 0.1509、3Sources 0.4608（24 格全部第一）。

#### 3.3.4 代码对照：公式 ↔ `alg/EF2FS.py`

| 论文公式 | 代码位置 | 对应关系 |
|---|---|---|
| 式(12) 视图权重 $c_i$ | L44–77 | 循环**外**先给每个视图建 `Lx_lst`（knn k=5、`t=1.0` 热核），循环**内**每轮算 `nu`；`energy>eps` 是除零保护（原实现无保护） |
| 式(21) $G$ 更新 | L86–88 | 分子三项 = ② `alpha*Y·B` + ③ `beta*Σx⁽ⁱ⁾w⁽ⁱ⁾` + ① `new_X·A`；分母 `(n_view*beta+1)*V` 对应 $(\beta V+1)G$ |
| 式(23) $B$ 更新 | L80 | `B ∘ YᵀG/(B·VᵀV)` |
| 式(25) $A$ 更新 | L90–94 | **分子/分母都用未加权的 `X`，而不是 `new_X`**（§5.2 H 行） |
| 式(27) $w^{(i)}$ 更新 | L96–102 | 分子 `beta*v[i]ᵀV + gamma*new_a`，`new_a = A[t1:t1+m[i], :]` 正是 $a^{(i)}$ |
| 式(28) 目标值 | L105–120 | `temp1..temp5`；`temp5 = Σ_p‖A_p‖_2` |

```python
# L62-77  每轮先算视图权重 c_i = 1/E_i 再归一化，E_i = Tr(GᵀL⁽ⁱ⁾G)
for i in range(n_view):
    energy = float(np.trace(V.T @ Lx_lst[i] @ V))
    nu_temp[i] = 1.0 / energy if energy > eps else 0.0    # ← 与式(12) 的 1/Tr(GᵀL⁽ⁱ⁾G) 一致
    nu_all = nu_all + nu_temp[i]
nu = nu_temp / nu_all
new_X = np.concatenate([v[i] * nu[i] for i in range(n_view)], axis=1)   # X^f

# L80-102  一轮里四个变量依次乘性更新
B = np.multiply(B, np.true_divide(np.dot(Y.T, V), np.dot(np.dot(B, V.T), V) + eps))

xw = np.dot(v[0], w[0])
for i in range(1, n_view):
    xw = xw + np.dot(v[i], w[i])                    # Σ_i x⁽ⁱ⁾w⁽ⁱ⁾

V = np.multiply(V, np.true_divide(                  # 代码的 V 就是论文的 G
    alpha * np.dot(Y, B) + beta * xw + np.dot(new_X, A),
    alpha * np.dot(np.dot(V, B.T), B) + (n_view * beta + 1) * V + eps))

Atmp = np.sqrt(np.sum(np.multiply(A, A), 1) + eps)
D = np.diag((0.5 / Atmp).flat)
A = np.multiply(A, np.true_divide(
    np.dot(X.T, V) + gamma * np.concatenate(w, axis=0),   # ← 注意是 X（未加权拼接）
    np.dot(np.dot(X.T, X), A) + gamma * A + lamb * np.dot(D, A) + eps))
```

**两处必须知道的实现差异**（详见 §5.2 EF²FS 行）：

1. **$A$ 的更新与目标函数用的不是同一个矩阵**：$G$ 的更新和目标值都用加权拼接 `new_X`，
   但 $A$ 的更新用 `X`（未加权）。于是 **$A$ 的更新方向并不是所记录目标函数的下降方向**
   （仅当所有 `nu[i]=1` 时才一致）。这是六篇里"目标值不可当收敛证据"的又一个实例（§5.7）。
2. **视图权重步在 $G$ 更新之前执行**，用的是**上一轮的 $G$**（滞后一轮），而式(12) 定义在当前 $G$ 上。

---

### 3.4 DHLI — 双层级混合标签识别（AAAI 2024）

> **一句话**：把观测标签矩阵**完全切分**成三块 —— 公共标签 $Y_c$、视图特有标签 $y_s^{(i)}$、噪声 $Y_n$，
> 把"去噪"从隐式正则（低秩、相关性过滤）变成**显式可学的变量**，并用一套离散逻辑算子
> （AND/OR/XOR）约束三块之间的关系。

**这篇的抓手：噪声从"被滤掉的干扰"变成"要求解的量"**

* 前人处理标签噪声的方式是隐式的：假设干净标签低秩、或按相关性排序把少数派丢掉
  （论文称之为"more is better than less"原则）。DHLI 说：这等于**没有定义什么是噪声**，
  所以只能启发式过滤，还会把真标签一起滤掉（FN）或留下噪声（FP），最终**扰乱特征排序**
  —— 这条"去噪误差 → 特征排序误差"的传导链是它最有洞察力的论证（见 P5）。
* 于是它引入 $Y_n$ 作为一等变量，并把 $Y$ 强制写成 $Y=Y_c\otimes Y_s\otimes Y_n$（逐元素 OR）。
  **"噪声"有了精确定义，优化器才能朝它走。**
* 这是六篇里**最早**的一篇（AAAI 2024），后续的 TOCL / I²VSLC 都沿用了这套标签分解。

#### 3.4.1 它认为前人错在哪

| # | 原文 | 中译 | 指向的缺陷 |
|---|---|---|---|
| P1 | "the basic assumption in traditional MVML methods is that the relevant label of each instance has been annotated **precisely**" | 传统方法假设标签被精确标注 | 忽略标签噪声 |
| P2 | "the resolution approach commonly employed is to utilize the principle of '**more is better than less**', whereby the minority is treated as noise, i.e., the low-rank technique, the filtering rule based on the order of the correlation level" | 常用"宁多勿少"原则，把少数派当噪声（低秩、相关性排序过滤） | **噪声缺乏精确定义** ⇒ 只能启发式过滤 |
| P3 | "the type of labels is divided into **two layers**, while establishing connections among **intra-layer and inter-layer** separately" | 标签分两层，分别建层内/层间连接 | 前人没研究"层级之间"的标签关系 |
| P4 | "due to the existing relationship between view-specific labels and observed labels, the dimension reduction for view-specific labels is **not enough for the direction of filtering noises**" | 仅对视图特有标签降维不足以为滤噪指明方向 | 降维 ≠ 去噪 |
| P5 | "The 'FP' relationship means that the noises still exist, and 'FN' means that the groundtruth label is filtered by mistake. Thus, the derivation **disturbs the order of features**" | FP 残留噪声、FN 误滤真标签 ⇒ 扰乱特征排序 | 去噪误差如何传导到特征排序 |

**P5 是最有洞察力的一条**：它把"标签去噪"与"特征排序"的**误差传导链**讲清楚了 ——
这是把两个通常分开讨论的问题接起来的论证。

#### 3.4.2 目标函数

**第一层（标签矩阵层）**，Definition 1：

$$
Y=Y_c\otimes Y_s\otimes Y_n,\qquad \otimes:\text{逐元素 OR}
\tag{3.16}
$$

**第二层（视图层）**，Definition 2：$Y_s=[y_s^{(1)},\dots,y_s^{(V)}]$，并施加

$$
\bigoplus_i y_s^{(i)}=0\ \text{(AND 必须为 0)},\qquad
\bigotimes_i y_s^{(i)}=Y_{tem}\approx Y_s\ \text{(OR 必须等于 }Y_s\text{)}
\tag{3.17}
$$

**完整目标**（式 12）：

$$
\begin{aligned}
\min\;&\sum_{i=1}^{V}\big\|X^{(i)}W^{(i)}-Y_c\big\|_F^2
+\sum_{i=1}^{V}\big\|X^{(i)}U^{(i)}-y_s^{(i)}\big\|_F^2\\
&+\alpha\underbrace{\|Y\ominus Y_c\|_F^2}_{\text{公共占多数}}
+\beta\underbrace{\|Y_n\|_1}_{\text{噪声稀疏}}
+\gamma\big(\|W\|_{2,1}+\|U\|_{2,1}\big)\\
&+\delta\Big(\underbrace{\big\|\circ_i y_s^{(i)}\big\|_F^2}_{\text{视图互斥 (f)}}
+\underbrace{\big\|\otimes_i y_s^{(i)}\ominus Y_s\big\|_F^2}_{\text{并集一致 (g)}}\Big)
\end{aligned}
\tag{3.18}
$$

其中 $\ominus$ 是逐元素 XNOR/XOR 型"不等"指示：

$$
y_{(pq)}\ominus y_{c(pq)}=
\begin{cases}1,& y_{(pq)}\ne y_{c(pq)}\\0,& y_{(pq)}=y_{c(pq)}\end{cases}
\tag{3.19}
$$

**先把四个算子认清楚**（这是读 DHLI 最容易卡住的地方）

| 算子 | 名字 | 在 $\{0,1\}$ 上的实际计算 | 语义 | 代码里的实现 |
|---|---|---|---|---|
| $\otimes$ | OR / 并集 | $\mathbf 1[\sum_i(\cdot)\ge1]$ | "至少一个视图说它是" | `sum_yt = 1[Σ y[i] > 0]` |
| $\circ$、$\bigoplus$、$\prod$ | AND / 交集 | $\prod_i(\cdot)$ | "所有视图都说它是"（全票通过） | `B = B * y[j]` |
| $\ominus$ | XOR | $\mathbf 1[a\ne b]$ | "不一致"指示器 → 惩罚项 | `tem = a - b; tem[tem==0]=0; tem[tem!=0]=1` |
| $\|Y_n\|_1$ | $\ell_1$ | $\sum_{pq}\lvert (Y_n)_{pq}\rvert$ | 噪声稀疏 | `2*np.sum(abs(Y_n))` |

关键技巧：**把不可微的逻辑约束"连续化"**。AND 写成**乘积**（连续、可导，其平方就是"全票通过"的罚项），
OR 写成**求和后取阈值**（不可导，但在乘性更新里它只作为**系数**出现，位置一旦定了就当常数），
XOR 写成**差的非零指示**。这样离散的标签组合逻辑就能进连续优化了。

**式(3.18) 的逐项解剖**

| 项 | 在说什么 | 为什么需要 |
|---|---|---|
| $\sum_i\|X^{(i)}W^{(i)}-Y_c\|_F^2$ | 每个视图都去拟合**同一个公共标签** $Y_c$ | 共性：所有视图共享的那部分 |
| $\sum_i\|X^{(i)}U^{(i)}-y_s^{(i)}\|_F^2$ | 每个视图去拟合**自己特有**的标签 $y_s^{(i)}$ | 互补：给每个视图一块"自留地"，避免共性把特色吞掉 |
| $\alpha\|Y\ominus Y_c\|_F^2$ | 公共标签应当**占多数**：$Y_c$ 解释不了的观测标签越少越好 | 防止模型把所有标签都推给 $Y_s$/$Y_n$（否则退化：谁都是特色、谁都是噪声） |
| $\beta\|Y_n\|_1$ | 噪声要**稀疏** | 噪声是少数派，$\ell_1$ 让它稀疏 |
| $\gamma(\|W\|_{2,1}+\|U\|_{2,1})$ | 两组权重都行稀疏 | 排序依据 |
| $\delta\|\circ_i y_s^{(i)}\|_F^2$ | **视图互斥**：$\prod_i y_s^{(i)}$ 统计的是"所有视图都声称这是自己的特色"的位置 | 一个对所有视图都"特有"的标签，按定义就是**公共**标签，不应留在 $Y_s$ 里（全票通过语义，不是软平均，§6.4 有量化） |
| $\delta\|\otimes_i y_s^{(i)}\ominus Y_s\|_F^2$ | **并集一致**：各视图特有标签的并集应当等于 $Y_s$ | 保证 $Y_s$ 被各视图"覆盖"完，不出现没人认领的标签 |

**变量清单**

| 符号 | 维度 | 角色 | 代码名 |
|---|---|---|---|
| $X^{(i)}$ | $n\times d^{(i)}$ | 已知特征 | `x[i]` |
| $Y$ | $n\times l$ | 已知观测标签 | `Y` |
| $Y_c$ | $n\times l$ | **要学**：公共标签 | `Y_c` |
| $y_s^{(i)}$ | $n\times l$ | **要学**：视图特有标签（$V$ 个） | `y[i]` |
| $Y_n$ | $n\times l$ | **要学**：噪声标签 | `Y_n` |
| $W^{(i)}$ | $d^{(i)}\times l$ | **要学**：拟合 $Y_c$ 的权重 | `w[i]` |
| $U^{(i)}$ | $d^{(i)}\times l$ | **要学**：拟合 $y_s^{(i)}$ 的权重 | `u[i]` |
| $Y_{tem}$ | $n\times l$ | **算出来**：各视图特有标签的 OR 聚合 | `sum_yt` |

**排序用的是两组权重之和**：论文 Algorithm 1 只写 $\|(W+U)_{(j)}\|_2$（$W,U$ 纵向拼接后求和），
代码先各自 MinMax 再相加（§5.2 M 行）。

维度自检：$X^{(i)}W^{(i)}=(n\times d^{(i)})(d^{(i)}\times l)=n\times l=\dim Y_c$ ✓（$U^{(i)}$ 同理）。

#### 3.4.3 算法

乘性更新，五变量轮流（式 16–20）：

$$
\begin{aligned}
Y_c &\leftarrow Y_c\circ\frac{A+\alpha Y+\delta B}{(V+\alpha+\delta)Y_c},&
A&=\sum_i X^{(i)}W^{(i)},\ B=Y\ominus Y_n\ominus Y_{tem}\\[2pt]
Y_n &\leftarrow Y_n\circ\frac{\delta C}{\beta QY_n+\delta Y_n},&
C&=Y\ominus Y_c\ominus Y_{tem},\ Q=\tfrac{1}{2|Y_n|}\\[2pt]
W^{(i)} &\leftarrow W^{(i)}\circ\frac{X^{(i)\top}Y_c}{X^{(i)\top}X^{(i)}W^{(i)}+\gamma D^{(i)}W^{(i)}}\\[2pt]
U^{(i)} &\leftarrow U^{(i)}\circ\frac{X^{(i)\top}y_s^{(i)}}{X^{(i)\top}X^{(i)}U^{(i)}+\gamma E^{(i)}U^{(i)}}\\[2pt]
y_s^{(i)} &\leftarrow y_s^{(i)}\circ\frac{X^{(i)}U^{(i)}+\delta H}{(1+\delta)y_s^{(i)}+\delta\,y_s^{(i)}\circ F\circ F}
\end{aligned}
\tag{3.20}
$$

**留一量**（式 20 下方）：

$$
F=\bigcirc_{j\ne i}y_s^{(j)},\quad
G=\bigotimes_{j\ne i}y_s^{(j)},\quad
H=Y\ominus Y_n\ominus Y_c\ominus G
\tag{3.21}
$$

**怎么读这组更新式**（乘性更新的通用推导见 §1.5.3）

* $W^{(i)}$ / $U^{(i)}$ 的更新最干净：$\dfrac{X^{(i)\top}(\text{目标标签})}{X^{(i)\top}X^{(i)}(\cdot)+\gamma E(\cdot)}$，
  就是**非负最小二乘**的一步乘性解 —— 分子是"特征与目标的相关"，分母是"当前重构 + 稀疏正则"。
  $D^{(i)}_{pp}=1/(2\|W_p\|_2)$、$E^{(i)}_{pp}=1/(2\|U_p\|_2)$ 是把 $\ell_{2,1}$ 辅助函数化之后的重加权对角阵：
  **当前行范数越小 ⇒ 惩罚越大**，这就是"迭代重加权"实现行稀疏的方式。
* $Y_c$ 的更新 $\dfrac{A+\alpha Y+\delta B}{(V+\alpha+\delta)Y_c}$ 是一次**证据表决**：
  $A=\sum_i X^{(i)}W^{(i)}$ 是各视图对公共标签的预测之和（$V$ 份证据）、$\alpha Y$ 把观测标签当先验计数、
  $\delta B$ 是"$Y\ominus Y_n\ominus Y_{tem}$"的修正项；分母把结果拉回与当前值同量级。
* $Y_n$ 的更新 $\dfrac{\delta C}{\beta QY_n+\delta Y_n}$ 里的 $Q=1/(2|Y_n|)$ 本意是**自适应 $\ell_1$ 重加权**，
  但代码把 $Y_n$ 硬二值化后 $Q\equiv1/2$，于是 $\beta QY_n$ 退化成常数 $\beta/2$（§5.2 M 行）。
* **留一量**：$F=\bigcirc_{j\ne i}y_s^{(j)}$（其他视图的 AND）、$G=\bigotimes_{j\ne i}y_s^{(j)}$（其他视图的 OR）、
  $H=Y\ominus Y_n\ominus Y_c\ominus G$。"留一"是为了**不让第 $i$ 个视图给自己投票**：
  约束"我不能与所有其他视图都声称特有的标签重复"，这就是视图互斥在更新式里的落地。

#### 3.4.4 创新点归因：为什么是 AAAI (CCF-A)

| 类别 | 创新点 | 论文证据 | 为什么够 A |
|---|---|---|---|
| **机制（主）** | 观测标签的**完全切分**为 hybrid labels（common / view-to-all specific / noisy），且"与噪声模式无关" | Abstract；Def.1；式(1)(12) | 引入**新数学对象** $Y_n$ 作为一等变量 ⇒ 判据 J1。前人把噪声当"要滤掉的干扰"，DHLI 把它当"要求的量" |
| **机制** | 引入**逻辑运算正则范式** $\otimes/\oplus/\ominus/\circ$，并把不可微的 AND 松弛为 Hadamard 积范数、OR 松弛为"求和+阈值" | 贡献 2；式(7)(9)(10)(11)(13)(14) | 把离散组合约束写进连续优化，且给了可行的松弛方案 |
| **机制** | **视图互斥**：$F=\bigcirc_{j\ne i}y_s^{(j)}$ 实现"一个标签不能对所有视图都特有" | 式(9)(10)(20) | 这是**全票通过语义**，非软平均（§6.4 有量化对比） |
| **建模** | 特征侧**双权重** $W^{(i)}\to Y_c$、$U^{(i)}\to y_s^{(i)}$ | 式(5)(6) | 一致性与互补性在同一损失内联合建模 |
| **优化** | 五变量乘性更新闭式解 | 式(15)–(20) | 属标准套路，**不构成 A 级贡献** |
| **实验** | 6 数据集 × 7 基线 × 4 指标 × 5 折 + 4 变体消融 + 4 参数敏感性 + 收敛曲线 | Table 1–4, Fig. 2–4 | 入场券 J3 |

**一句话定位**：DHLI 上 AAAI 靠的是 **J1（hybrid labels 是新数学对象）+ 逻辑正则的松弛技巧 + 误差传导链论证**，不是性能。

#### 3.4.5 实验证据（数字已按 PDF 坐标校正）

| 指标 | SCENE | OBJECT | MIRFlickr | Corel5K | IAPRTC12 | 3Sources |
|---|---|---|---|---|---|---|
| AP ↑ | 0.7951±0.012 | 0.4845±0.046 | 0.6635±0.015 | 0.2370±0.009 | 0.1421±0.007 | 0.4534±0.034 |
| Coverage ↓ | 0.4174 | 0.2979 | 0.5897 | 0.4832 | 0.5091 | 0.6181 |
| HL ↓ | 0.0969 | 0.0572 | 0.1808 | 0.01379 | 0.01549 | 0.2315 |
| RL ↓ | 0.0926 | 0.1642 | 0.1671 | 0.2283 | 0.2068 | 0.4761 |

**优势幅度（必须诚实指出）**：AP 上 SCENE 领先次优 MRDM 仅 $+0.0022$、Corel5K 领先 M2LD 仅 $+0.0015$；Coverage 在 OBJECT 上领先 CLML 仅 $0.0001$、IAPRTC12 领先 ELSMML 仅 $0.0003$；HL 在三个数据集上的领先量级是 $10^{-4}\sim10^{-5}$。**多数差距落在标准差内。**

**消融（Table 4, AP）**：去掉 $\delta$（视图特有约束）掉点最大（SCENE $0.7951\to0.7842$，$-1.37\%$），而去掉 $\alpha/\beta/\gamma$ 各只掉 $0.001$ 量级。**只有视图互斥项真正关键** —— 这与 §3.4.4 把"视图互斥"列为第二机制创新点一致。

#### 3.4.6 代码对照：公式 ↔ `alg/DHLI.py`

| 论文公式 | 代码位置 | 对应关系 |
|---|---|---|
| 式(16) $W^{(i)}$ | L71–72 | 分子 `x[i]ᵀ Y_c`；分母 `x[i]ᵀx[i]w[i] + gamma*D*w[i]` |
| 式(17) $U^{(i)}$ | L73–74 | 分子 `x[i]ᵀ y[i]`；分母用 `E = diag(1/(2‖U_p‖))` |
| 式(18) $Y_c$ | L114–118 | `tem2 = Y − Y_n − sum_yt` 的 $\ominus$ 指示；分子 `t1 + alpha*Y + lamb*tem2`（`t1 = Σ_i x[i]w[i]` 即 $A$） |
| 式(19) $Y_n$ | L127–132 | `tem3 = Y − Y_c − sum_yt`；`Q = 1/(2*abs(Y_n))` |
| 式(20) $y_s^{(i)}$ | L76–93 | 留一量 `y_o`（其他视图 OR）与 `B`（其他视图 AND） |
| 式(12) 目标值 | L144–184 | `temp1..temp8`（`temp4 = 2β‖Y_n‖₁`，多一个因子 2） |
| 排序 | L202–212 | `A = MinMax(W) + MinMax(U)`，按 $\|A\|_2$ 降序 |

```python
# L76-93  留一量 + y_s⁽ⁱ⁾ 的乘性更新
y_o = sum_yt - y[i]
y_o[y_o == 0] = 0
y_o[y_o > 0] = 1           # y_o = OR_{j≠i} y[j]：其他视图的并集

B = 1
for j in range(n_view):
    if i != j:
        B = B * y[j]       # B = ∏_{j≠i} y[j] = AND（全票通过语义）
        # 原实现写成 B = 0 再 B = B * y[i]，恒为 0 ⇒ 互斥项静默失效；此仓库已修正

tem1 = Y_spe - y_o
tem1[tem1 == 0] = 0
tem1[tem1 > 0] = 1
tem1[tem1 < 0] = 1         # tem1 就是 ⊕ 指示器
y[i] = np.multiply(y[i], np.true_divide(
    np.dot(x[i], u[i]) + lamb * tem1,
    y[i] + lamb * y[i] + lamb * y[i] * B * B + eps))   # ← lamb*y[i]*B*B 即视图互斥项
```

分母里的 `lamb*y[i]*B*B` 对应式(20) 的 $\delta\,y_s^{(i)}\circ F\circ F$（乘积出现两次，因为罚项是平方），
分子里的 `lamb*tem1` 对应并集一致项；`(1+δ)y_s` 就是 `y[i] + lamb*y[i]`。

```python
# L98-140  每轮对 y[i]、Y_c、Y_n 各做一次 MinMax + 0.5 硬二值化 —— 论文完全没有这一步
y[i] = MinMax(y[i]);  y[i][y[i] <= 0.5] = 0;  y[i][y[i] > 0.5] = 1
Y_c  = MinMax(Y_c);   Y_c[...] = 0/1
Y_n  = MinMax(Y_n);   Y_n[...] = 0/1
```

这一步的三个后果（§5.2 H 行）：① 变量被强制成 $\{0,1\}$，而且 MinMax 会让**每列必然至少有一个 1**；
② $0$ 是乘性更新的吸收态，被压成 0 的元素再也回不来；③ 在乘性更新之间插入硬投影，
**破坏论文所声称的收敛前提**（而论文贡献点里恰好写着 "with **proven** convergence"）。

另外两处已知偏差：`Y_spe` 的语义前后不一致（初始化把负差归 0、循环内归 1），
以及目标值里 $\beta$ 项写成 `2β‖Y_n‖₁` —— 见 §5.2 的 DHLI 表。

---

### 3.5 GRAFS — 锚点引导的全局视图重构（Inf. Sci. 2024）

> **一句话**：把"全局视图"从一个**假设**（把各视图特征直接拼起来）变成一个**待解变量** $X^f$，
> 再用一个**锚点空间**（$n$ 个实例 × $k$ 个锚点）把"候选视图 $B$"与"全局视图 $X^f$"钉在**同一套簇坐标**里，
> 使共享/特有信息的比例可以连续调节。

**这篇的抓手：全局视图是学出来的**

* 多视图特征选择绕不开一个问题：最终要在**所有视图**上排一个统一的特征序。前人要么把各视图特征
  直接拼接（噪声随维度一起进来，"哪块属于哪个视图"的结构也被抹掉），要么假设它们共享一个低维子空间
  （连视图特有的差异一起压掉）。
* GRAFS 的做法是**学**一个全局视图 $X^f$，让它同时被三方牵引：标签（$X^fW\approx Y$）、
  各视图（$X^fD^{(i)}\approx X^{(i)}C^{(i)}$）、以及一个共享的**锚点空间**。
* **"锚点"到底指什么**：$B\in\mathbb R^{n\times k}$ 是"候选视图"，可理解为"用 $k$ 个锚点（$k$ 个代表性实例/簇中心）
  来表达这 $n$ 个样本"的一份潜表示。两条路径都通过**同一个** $W^c\in\mathbb R^{n\times k_1}$（簇成员矩阵）
  绕到锚点空间 —— 这就是它区别于"简单拼接"的实质，也是 §3.5.3 要讲的那处设计。

#### 3.5.1 它认为前人错在哪

| # | 原文 | 中译 |
|---|---|---|
| Q1 | "feature weights derived solely from individual views necessitate **additional constraints** to handle scenarios involving overlapping or missing information" | 仅从单视图导出的权重缺少约束 |
| Q2 | "Zhu et al. devise mappings for the global-view and local-view, with the global-view synthesized by **concatenating the original features**. Regrettably, the oversight of noises and relationships within the feature space can impede the accurate estimation" | 全局视图 = 原始拼接 ⇒ 噪声随维度进入 + 切分不清 |
| Q3 | "it remains imperative to explore the **distinct characteristics** of each view to effectively capture global-view information" | 共享子空间抹掉了视图特有成分 |
| Q4 | "Instead of simply concatenating the views, we need to consider the relationships among them… takes into account **both the view-relationship and the instance-relationship**" | 需同时建模视图关系与实例关系 |

**Q2 是核心指控**：全局视图在 GRAFS 之前是一个**假设**（拼接或共享子空间投影），
而不是从数据中**学**出来的量。

#### 3.5.2 双路径多层分解

$$
L_E=\underbrace{\sum_{i=1}^{V}v_i\big\|X^{(i)}-BP^{(i)}\big\|_F^2}_{\text{锚点重构各视图}}
+\underbrace{\big\|B-W^cA^cR^c\big\|_F^2}_{\text{候选视图路径}}
+\underbrace{\big\|X^f-W^cA^fR^f\big\|_F^2}_{\text{全局视图路径}}
+\underbrace{\big\|A^c-A^f\big\|_F^2}_{\text{锚点引导一致性}}
\tag{3.22}
$$

**完整目标**（式 14）：

$$
L_{GRAFS}=\underbrace{\|X^fW-Y\|_F^2}_{(a)}+\alpha\underbrace{\sum_i v_i\|X^{(i)}-BP^{(i)}\|_F^2}_{(b)}
+\beta L_E^{(\text{c})}+\gamma\underbrace{\sum_i v_i\|X^fD^{(i)}-X^{(i)}C^{(i)}\|_F^2}_{(d)}
+\delta\underbrace{\|W\|_{2,1}}_{(e)}
\tag{3.23}
$$

维度自检（这组分解能成立的关键）：

$$
W^cA^cR^c=(n\times k_1)(k_1\times k_1)(k_1\times k)=n\times k\Rightarrow\text{还原 }B
$$
$$
W^cA^fR^f=(n\times k_1)(k_1\times k_1)(k_1\times d)=n\times d\Rightarrow\text{还原 }X^f
$$

> **一处容易搞错的维度**：全局视图 $X^f$ 是 $n\times d$（**与拼接后的特征同维**），不是 $n\times k$。
> 因为最终排序矩阵 $W$ 必须是 $d\times l$（$d$ 行才能给 $d$ 个特征排序），而 $X^fW$ 要等于 $n\times l$ 的 $Y$。
> 论文的 Lagrangian 乘子表（$\omega\in\mathbb R^{n\times d}$ 对应 $X^f$）与代码一致；
> 只有 $B$（候选视图）是 $n\times k$，其中 $k$ 是**锚点数**（代码 `kk`，默认 20）。

**变量清单（$k$ = 锚点数，$k_1$ = 潜锚点维数，$d$ = 总特征数）**

| 符号 | 维度 | 角色 | 代码名 |
|---|---|---|---|
| $X^{(i)}$ | $n\times d^{(i)}$ | 已知特征 | `x[i]` |
| $B$ | $n\times k$ | **要学**：候选视图（锚点表达） | `B` |
| $X^f$ | $n\times d$ | **要学**：全局视图（潜变量） | `X1` |
| $W$ | $d\times l$ | **要学**：全局视图上的特征权重 = **排序依据** | `W` |
| $W^c$ | $n\times k_1$ | **要学**：两条路径**共享**的簇成员矩阵 | `W1` |
| $A^c,A^f$ | $k_1\times k_1$ | **要学**：两条路径各自的潜锚点 | `A1`, `A2` |
| $R^c$ / $R^f$ | $k_1\times k$ / $k_1\times d$ | **要学**：锚点→候选视图 / 锚点→全局视图 的系数 | `R1`, `R2` |
| $P^{(i)}$ | $k\times d^{(i)}$ | **要学**：用锚点重构第 $i$ 个视图 | `p[i]` |
| $C^{(i)}$ | $d^{(i)}\times d^{(i)}$ | **要学**：视图内的特征变换（代码**强制对角**） | `s[i]` |
| $D^{(i)}$ | $d\times d^{(i)}$ | **不更新**：把 $X^f$ 的列块切出来的选择矩阵 | `d[i]` |
| $v_i$ | 标量 | **算出来**：视图权重 | `nu[i]` |

**逐项解剖**（式 3.23）

| 项 | 在说什么 | 为什么需要 |
|---|---|---|
| (a) $\|X^fW-Y\|_F^2$ | 学出来的全局视图要能**预测标签** | 这是唯一把 $X^f$ 与监督信号连起来的项，也是排序矩阵 $W$ 的来源 |
| (b) $\alpha\sum_i v_i\|X^{(i)}-BP^{(i)}\|_F^2$ | 候选视图 $B$ + 各视图自己的系数 $P^{(i)}$ 要能**重构每个视图** | 保证 $B$ 承载了各视图的信息（它是在各视图上"学"出来的，不是拼出来的） |
| (c) $\beta L_E$ | 两条分解路径 + 软一致性（式 3.22） | 见 §3.5.3，这是全文最精妙的一处 |
| (d) $\gamma\sum_i v_i\|X^fD^{(i)}-X^{(i)}C^{(i)}\|_F^2$ | 全局视图的**第 $i$ 个列块**要能通过 $C^{(i)}$ 还原第 $i$ 个视图 | 让全局视图逐视图"对得上"，$D^{(i)}$ 把"按视图切块"变成一个可微的矩阵乘法 |
| (e) $\delta\|W\|_{2,1}$ | 行稀疏 | 排序 |

**两条路径在读法上的一句话版本**：

```
                     B（候选视图，n×k）── 共享同一套簇坐标 W^c ──► X^f（全局视图，n×d）
                     │                                              │
        用锚点重构各视图 (b)                            预测标签 (a) + 逐视图还原 (d)
```

#### 3.5.3 为什么必须"共享 $W^c$"且"$A^c\approx A^f$（不是相等）"

这是 GRAFS 最精妙的一处设计，也是它区别于"简单拼接"的实质：

* **共享 $W^c$（硬共享"上下文"）**：$W^c$ 是实例侧的软聚类成员矩阵。若两条路径各用各的 $W$，
  则 $B\approx W_1A^cR^c$ 与 $X^f\approx W_2A^fR^f$ 只在**相差任意可逆旋转/尺度**的意义下成立
  （非负分解的尺度-置换不可辨识性），此时令 $A^c\approx A^f$ 可被**重新参数化白白满足**，
  约束失去意义。共享 $W^c$ 把两个分解钉在**同一实例簇坐标**中。
* **$A^c\approx A^f$（软一致而非硬相等）**：消融 GRAFS$_{ver2}$（硬等同，式 29）的证据：
  结构清晰的数据集上几乎无损（yeast 0.6608 vs 0.6609、SCENE 同为 0.7977），
  但确有差异的数据集上明显退化（OBJECT 0.4859 vs 0.4976、VOC07 0.5856 vs 0.5913、
  3Sources 0.4687 vs 0.4718）。软约束允许 $A^f$/$R^f$ 保留差异以吸纳视图特有信息。

**降噪的副作用**：两级压缩 $n\to k_1\to k$ 再展开到 $d$ 本身是一次低通滤波。
原文："During the process of compressing and expanding dimensions, the noise is filtered out"。

#### 3.5.4 创新点归因：为什么是 Inf. Sci.

| 类别 | 创新点 | 证据 |
|---|---|---|
| **机制（主）** | **潜锚点空间里的软一致性**替代硬共享子空间：用共享 $W^c$ 把 $B$ 与 $X^f$ 投到同一簇坐标，再约束 $A^c\approx A^f$，使共享/特有信息**连续可调** | 式(7) 第 3 项；消融 Table 6 |
| **建模（主）** | **全局视图是学出来的潜变量**，同时被标签（$X^fW\!\approx\!Y$）、锚点（$W^cA^fR^f$）、各视图（$X^{(i)}C^{(i)}D^{(i)\top}$）三方牵引 | 论文 §3.1；式(14)(27) |
| **机制** | 视图关系与实例关系**解耦为两条通道**：$v_i$（谁更重要）与 $P^{(i)}$（实例如何映射） | 式(5)(6)；论文 §3.2 |
| **机制** | 用切分算子 $D^{(i)}$ 把"按视图切块"变成**可微矩阵乘积** | 式(10)(11)(12) |
| **优化** | 十变量全部乘性闭式解，每迭代 $O(d^2n)$（对 $n$ 线性） | 式(18)–(27)；论文 §3.6 |

**为什么只到 B**：
1. **锚点数 $k$ 与聚类数 $k_1$ 从未做敏感性分析** —— 论文只报 $\alpha,\beta,\gamma,\delta$ 的敏感性
   （Fig.3），而 $k,k_1$ 是模型**结构参数**。全文与 $k$ 相关的证据只有 GRAFS$_{ver1}$（令 $k=d$）的退化消融。
   代码里 $k_1=10$ 硬编码、$kk$ 默认 20（注释提示曾用 100）—— **论文中找不到这三个数字**。
2. **相对 [6][9] 是组合式增量**而非新范式；
3. 式(17) 与式(27) 各有一处笔误（代码修对了）；
4. Table 3 与 Table 6/7 对同一模型给出**三个不同的 OBJECT AP 值**（0.4987 / 0.4976 / 0.4850），论文未说明。

**实验数字（AP ↑）**：yeast 0.6609、SCENE 0.7977、OBJECT 0.4987、VOC07 0.5913、
MIRFlickr 0.6795、3Sources 0.4718（6 数据集 4 指标全部第一）。
**但 yeast 上相对 MoRE 的优势是 Coverage 0.6356 vs 0.6357、HL 0.2254 vs 0.2259 —— 统计意义存疑。**

**运行时间（Table 5, 秒）**：GRAFS 在 OBJECT 86.887 vs MRDM 2360.426（约 27×），
但在 3Sources（$d{=}3000\gg n{=}169$）上 28.203 **反而慢于** M2LD(0.745)/SSFS(2.258)/MRDM(5.995)
—— 当 $d\gg n$ 时 $O(d^2n)$ 中的 $d^2$ 主导，锚点的收益被抵消。论文未讨论这一 regime。

**求解**：十个变量全部有乘性闭式解（通用推导见 §1.5.3），每迭代 $O(d^2n)$；但结构参数 $k_1$（硬编码 10）与 $kk$（默认 20）论文都没报（§5.2 GRAFS 行）。

#### 3.5.5 代码对照：公式 ↔ `alg/GRAFS.py`

| 论文公式 | 代码位置 | 对应关系 |
|---|---|---|
| 式(2)(3)(5) 视图亲和图 / $v_i$ | L79–96 | **建在标签矩阵上**（`construct_W(Y_ori, k=20, t=1.0)`），不是论文要求的视图特征图；`nu[i] = 1/Tr(x[i]ᵀ L_y x[i])`（§5.2 H 行） |
| 式(25) $W$ | L112–115 | 分子 `X1ᵀY`、分母 `X1ᵀX1W + lamb·E·W`，`E = diag(0.5/‖W_p‖)` |
| 式(18)(19)(27) $R^c,R^f,W^c$ | L118–122 | 两条路径的分解与共享 |
| 式(29) 锚点软一致性 $\|A^c-A^f\|^2$ | L120–121 | **分子里的 `+A2` / `+A1` 就是这一项的梯度** |
| 式(14) $B$ | L135 | 分子 `alpha*q1 + beta*W1A1R1`，`q1 = Σ_i nu[i]·x[i]P^{(i)ᵀ}` |
| 式(14) $X^f$ | L137–138 | 分子 = (a) `YWᵀ` + (c) `beta·W1A2R2` + (d) `gamma*q3` |
| $C^{(i)}$ / $D^{(i)}$ | L59–75 | `s[i]` 是**对角阵**（论文是满矩阵）；`d[i]` 是 $d\times d^{(i)}$ 选择矩阵 |
| 式(14) 目标值 | L140–155 | `temp1..temp5` |

```python
# L46-56  结构参数与初始化（k1 与 kk 这两个数字在论文里找不到）
k1 = 10                                    # 潜锚点维数，硬编码
B = rng.random((num, kk))                  # 候选视图 B ∈ R^{n×k}；kk 默认 20（注释提示曾用 100）
W = rng.random((feature_num, label_num))   # W ∈ R^{d×l}
A1 = rng.random((k1, k1)); A2 = rng.random((k1, k1))    # A^c, A^f
W1 = rng.random((num, k1))                 # W^c ∈ R^{n×k1}
X1 = normalization(np.dot(Y, W.T))         # X^f 用 YWᵀ 暖启动（论文未提）
```

```python
# L91-96  视图权重：建在标签图 Ly 上（论文要的是视图特征图）
Sy = dense(construct_W(Y_ori, **options))  # options: knn k=20, heat_kernel, t=1.0
Ly = Ay - Sy
for i in range(n_view):
    nu_temp[i] = 1 / np.trace(x[i].T @ Ly @ x[i])
nu = nu_temp / nu_all
```

```python
# L118-122  两条路径的分解 + 共享 W^c + 锚点软一致性
R1 = R1 * (A1.T @ W1.T @ B) / (A1.T @ W1.T @ W1 @ A1 @ R1 + eps)      # R^c：锚点 → 候选视图
R2 = R2 * (A2.T @ W1.T @ X1) / (A2.T @ W1.T @ W1 @ A2 @ R2 + eps)     # R^f：锚点 → 全局视图
A1 = A1 * (W1.T @ B @ R1.T + A2) / (W1.T @ W1 @ A1 @ R1 @ R1.T + A1 + eps)   # ← +A2
A2 = A2 * (W1.T @ X1 @ R2.T + A1) / (W1.T @ W1 @ A2 @ R2 @ R2.T + A2 + eps)  # ← +A1
W1 = W1 * (B @ R1.T @ A1.T + X1 @ R2.T @ A2.T) \
        / (W1 @ A1 @ R1 @ R1.T @ A1.T + W1 @ A2 @ R2 @ R2.T @ A2.T + eps)
```

`A1`/`A2` 的更新里各出现对方的项，这就是 $\|A^c-A^f\|_F^2$ 的梯度落到代码里的样子；
`W1` 的分子同时含两条路径的项，说明 **$W^c$ 是硬共享的**（不是两份参数）。

```python
# L135-138  B 与 X^f 的更新（q1..q4 已按 v_i = nu[i] 累加过各视图的贡献）
B = np.multiply(B, np.true_divide(alpha * q1 + beta * np.dot(np.dot(W1, A1), R1),
                                  alpha * q2 + beta * B + eps))
X1 = np.multiply(X1, np.true_divide(
    np.dot(Y, W.T) + beta * np.dot(np.dot(W1, A2), R2) + gamma * q3,   # (a) + (c) + (d)
    np.dot(np.dot(X1, W), W.T) + beta * X1 + gamma * q4 + eps))
```

**读这段代码要带上 §5.2 的 GRAFS 四条**：视图亲和图被换成**标签图**（语义转置）、
$C^{(i)}$ 被强制对角（参数量从 $O(d^{(i)2})$ 降到 $O(d^{(i)})$）、$k_1=10$ 与 $kk=20$ 论文未报告、
非负性没有任何强制（数据含负值时乘性更新可能发散）。另外 L22 的注释写着"before setting is 100"，
说明 `kk` 这个结构参数在开发过程中改过 —— 论文里对此没有任何交代。

---

### 3.6 I²VSLC — 视图特有标签关系（Inf. Sci. 2024）

> **一句话**：打破"所有视图共用同一套标签"的假设 —— 每个视图学一份**自己的**标签 $y^{(i)}$，
> 观测标签只是它们的**并集** $Y_{vs}=\mathbf 1[\sum_i y^{(i)}\ge1]$；再用"视图内流形平滑 + 视图间共识"
> 这**两层**（方法名 $I^2$ 的来历）把 $V$ 份标签绑在一起。

**这篇的抓手：标签空间也会"不一致"**

* 前人注意力都在**视图空间**（怎么把多个视图的特征融合好），默认标签空间是统一的：所有视图对应同一个 $Y$。
  I²VSLC 指出这在非对齐 / 标注不全的场景下不成立，而且"统一标签"会逼某些视图去拟合根本不属于它的标签，
  把互补信息抹平。
* 它的解法是给每个视图一份可学标签 $y^{(i)}$，并且**只约束它们的并集等于观测标签**
  —— 这样"某标签只在部分视图里出现"就成了模型可以表达的东西。
* 但注意它**没有**做"视图特有的相关矩阵"：标签相关矩阵 $C$ 仍是全局共享的 $l\times l$（见 3.6.1 的澄清框）。

#### 3.6.1 它认为前人错在哪

论文的批判分散在六处，可归纳为：

| # | 原文 | 中译 | 核心指控 |
|---|---|---|---|
| R1 | "the inconsistency in **label space** has been underexplored, revealing the inadequacy of assuming uniform labels" | 标签空间的**不一致性**未被充分探索 | 现有工作只在**视图空间**平衡共性与互补 |
| R2 | "these methods assume an ideal **uniform** label matrix, thus limiting their practical applicability" | 假设理想统一标签矩阵，限制实用性 | 统一标签假设不成立 |
| R3 | "the assumption is **inappropriate** that all views have a uniform set of labels" | "所有视图标签集相同"不恰当 | 非对齐场景 |
| R4 | "the matrix transformation remains rooted in the **shared** aspects of the learning process" | 高阶相关法构造的相关矩阵仍**共享** | 批 [8][38] |
| R5 | "insufficient attention has been given to the **varying levels of correlation** between labels" | 标签相关性的**层次差异**未受重视 | 批 [39]（虽用 view-specific labels） |

> **⚠️ 一处必须澄清的措辞**：论文批判的是**统一标签矩阵**，而它自己学到的标签相关矩阵
> $C$ 仍是**全局共享**的一个 $l\times l$ 矩阵（式 (9)(11)(24) 中只有一个 $C$）。
> "视图特有"完全由 ①每视图的 $y^{(i)}$、②每视图的图 $L^{(i)}$、③视图间两两共识承载。
> **不要把这篇说成"视图特有相关矩阵"。**

#### 3.6.2 自刻画函数 $\Phi(\cdot)$

论文式 (1) 把目标拆成五项，第二项即"自画像"：

$$
\Phi(\cdot)=\underbrace{\sum_{i=1}^{V}\mathrm{Tr}\big(y^{(i)\top}L^{(i)}y^{(i)}\big)}_{\text{intra-view 视图内流形平滑}}
+\underbrace{\sum_{i=1}^{V}\sum_{j\ne i}\big\|y^{(i)}-y^{(j)}\big\|_F^2}_{\text{inter-view 视图间共识}}
\tag{3.24}
$$

**$\Phi$ 的两项各有讲究**：

* **intra-view**：$L^{(i)}=A^{(i)}-S^{(i)}$ 的图建在**该视图的特征空间**（式 (3) 的距离取 $\|x_i^{(v)}-x_j^{(v)}\|$），作用在**标签**上。由平滑假设
  $\frac12\sum_{i,j}s_{ij}\|y_i-y_j\|^2=\mathrm{Tr}(y^\top Ly)$ 得到。
  *这是"视图特有"的唯一来源* —— $L^{(i)}$ 只编码第 $i$ 个视图的实例结构。
  **（仓库笔记 `PAPERS_AND_CODE.md` §5 写作"标签图/标签空间建图"，属文档表述错误。）**
* **inter-view**：刻意**不用均值/标准参考点**。原文：
  "In order to mitigate any potential impact resulting from the quality of a specific reference point such as **average value**, a standard point is **not** employed"。
  这是一个有意的设计选择 —— 避免"平均模板本身质量差"污染共识。

#### 3.6.3 目标函数与算法

$$
\begin{aligned}
F_{I^2VSLC}=&\sum_{i}\big\|x^{(i)}w^{(i)}-y^{(i)}\big\|_F^2
+\alpha\Big(\sum_i\mathrm{Tr}(y^{(i)\top}L^{(i)}y^{(i)})+\sum_i\sum_{j\ne i}\|y^{(i)}-y^{(j)}\|_F^2\Big)\\
&+\beta\big(\|YC-Y_{vs}\|_F^2+\|Y-YC\|_F^2\big)
+\gamma\sum_i\|w^{(i)}\|_F+\delta\|W\|_{2,1}
\end{aligned}
\tag{3.25}
$$

其中 $Y_{vs}=\mathbf{1}[\sum_i y^{(i)}\ge1]$（式 15 的 OR 聚合，$\otimes$ 被论文显式定义为 OR 而非 Kronecker 积）。

**层级特征选择**（式 12 + 式 13）：$\gamma\sum_i\|w^{(i)}\|_F$ 是**视图级**组收缩，
$\delta\|W\|_{2,1}$ 是**特征级**行稀疏。论文的语义总结：
"informative features will be selected from the **important views** as much as possible"。

**乘性更新**（式 21/24/27）：

$$
\begin{aligned}
w^{(i)} &\leftarrow w^{(i)}\circ\frac{x^{(i)\top}Y}{x^{(i)\top}x^{(i)}w^{(i)}+\gamma E^{(i)}w^{(i)}+D^{(i)}w^{(i)}}\\
C &\leftarrow C\circ\frac{Y^\top Y_{vs}+Y^\top Y}{2Y^\top YC}
\quad\text{（仓库实现的是这个 KKT 正确版）}\\
y^{(i)} &\leftarrow y^{(i)}\circ\frac{\alpha S^{(i)}y^{(i)}+\alpha(V{-}1)F+x^{(i)}w^{(i)}+\beta(YC+G)}
{\alpha A^{(i)}y^{(i)}+\alpha(V{-}1)y^{(i)}+(1+\beta)y^{(i)}}
\end{aligned}
\tag{3.26}
$$

其中 $F=Y_{vs}-y^{(i)}$、$G=y^{(i)}-Y_{vs}$。

**变量清单**

| 符号 | 维度 | 角色 | 代码名 |
|---|---|---|---|
| $x^{(i)}$ | $n\times d^{(i)}$ | 已知特征块 | `v[i]` |
| $Y$ | $n\times l$ | 已知观测标签 | `Y` |
| $y^{(i)}$ | $n\times l$ | **要学**：第 $i$ 个视图特有的标签 | `y[i]`（随机二值初始化） |
| $w^{(i)}$ | $d^{(i)}\times l$ | **要学**：视图自己的特征权重 | `w[i]` |
| $W=\bigoplus_i w^{(i)}$ | $d\times l$ | 排序依据（纵向拼接） | `B = np.concatenate(w, axis=0)` |
| $C$ | $l\times l$ | **要学**：标签相关矩阵（**全局共享**） | `AA`（初始化为单位阵） |
| $Y_{vs}$ | $n\times l$ | **算出来**：各视图标签的 OR | `sum_Yt` |
| $L^{(i)}$ | $n\times n$ | 已知：视图 $i$ 的**特征** knn 图拉普拉斯（k=5、`t=1.0`） | `Lx_lst[i]` |

**逐项解剖**（式 3.25）

| 项 | 在说什么 | 为什么需要 |
|---|---|---|
| ① $\sum_i\|x^{(i)}w^{(i)}-y^{(i)}\|_F^2$ | 每个视图用自己的权重拟合**自己的**标签 | 互补性的来源。注意实现里分子用的是观测 $Y$ 而不是 $y^{(i)}$（论文自身的笔误被代码沿用，§5.2 M 行） |
| ②a $\alpha\sum_i\mathrm{Tr}(y^{(i)\top}L^{(i)}y^{(i)})$ | **intra-view**：在视图 $i$ 的**特征图**上，标签要平滑（特征相近 ⇒ 标签相近） | "视图特有"的唯一来源；$L^{(i)}$ 只编码第 $i$ 个视图的实例结构 |
| ②b $\alpha\sum_i\sum_{j\ne i}\|y^{(i)}-y^{(j)}\|_F^2$ | **inter-view**：各视图的标签两两互相靠近 | 共识项；刻意不用均值参考点，避免"平均模板本身质量差"污染共识 |
| ③ $\beta(\|YC-Y_{vs}\|_F^2+\|Y-YC\|_F^2)$ | $YC$ 要同时接近"各视图标签的并集"$Y_{vs}$ 与观测 $Y$ | 用标签相关矩阵 $C$ 做**增强标签**去噪；两个方向缺一不可 |
| ④ $\gamma\sum_i\|w^{(i)}\|_F$ | **视图级**组收缩 | 让不重要的视图整块变小 ⇒ "从重要视图里选特征" |
| ⑤ $\delta\|W\|_{2,1}$ | **特征级**行稀疏 | 排序 |

**"层级特征选择"的含义**：④ 与 ⑤ 是**两个粒度**的稀疏 —— ④ 以"整个视图的权重矩阵"为单位收缩（Frobenius 范数），
⑤ 以"某个特征在所有视图里的行"为单位收缩。合起来就是论文那句话：
"informative features will be selected from the **important views** as much as possible"。

**维度自检**：$x^{(i)}w^{(i)}=n\times l$ ✓；$YC=(n\times l)(l\times l)=n\times l$ ✓；$Y_{vs}=\mathbf 1[\sum_i y^{(i)}\ge1]$ 仍是 $n\times l$ ✓。

**怎么读这组更新式**（式 3.26；乘性更新的通用推导见 §1.5.3）

* $w^{(i)}$：分子 $x^{(i)\top}Y$（"特征与标签的相关"），分母里**同时出现两个对角阵** ——
  $\gamma E^{(i)}$（视图级，$E$ 是标量×I 的松弛）与 $\delta D^{(i)}$（特征级，$D_{pp}=1/(2\|W_p\|_2)$ 逐行重加权）。
  这两个对角阵就是"层级稀疏"在代数上的痕迹。
* $C$：分子 $Y^\top Y_{vs}+Y^\top Y$，分母 $2Y^\top YC$ —— 仓库实现的是 KKT 正确版
  （论文印刷版式(23) 把 $Y^\top Y$ 的符号写反了，§5.2 M 行）。
* $y^{(i)}$：分子四项对应"视图内平滑（$\alpha S^{(i)}y^{(i)}$）+ 与其他视图一致（$\alpha(V{-}1)F$）
  + 拟合特征（$x^{(i)}w^{(i)}$）+ 与增强标签一致（$\beta(YC+G)$）"，分母是它们各自的抑制项。

#### 3.6.4 创新点归因：为什么是 Inf. Sci.（CCF-B / 中科院一区 Top）

| 类别 | 创新点 | 证据 |
|---|---|---|
| **机制（主）** | 显式引入**视图特有标签** $y^{(i)}$ 作为待优化变量；把观测 $Y$ 建模为各视图标签的**并集** | §1"view-specific labels are defined"；$Y_{vs}=\mathbf{1}[\sum y^{(i)}\ge1]$ | 
| **机制** | **intra-view + inter-view 双层**标签关系（方法名 $I^2$ 即由此而来），且刻意不用均值参考点 | 式 (5)(6)(7) |
| **机制** | 动态标签相关矩阵 $C$ 做**增强标签**去噪 | 式 (9)(11)；Table 1 唯一在 "View-specific labels" 与 "Label noises" 同时打勾 |
| **建模** | 层级特征选择：视图级组收缩 + 特征级行稀疏 | 式 (12)(13) |
| **实验** | 6 数据集（含 $n{=}6047$、$l{=}268$、$d{=}3000$）× 7 基线 × 4 指标 | Table 2–4 |

**为什么只到 B 不到 A**：
1. **无独立消融实验**（全文无 "ablation" 字样，无 Table 5 / Fig 5）—— 这是相对 DHLI 最明显的短板；
2. 乘性更新是**标准 NMF/KKT 模板**，且论文章节 4 的更新式（式 23/24）有符号级笔误（代码实现了正确版）；
3. 核心抓手（视图特有标签）在 [39] 已有雏形，本文是**推广而非原创**；
4. 标签相关矩阵 $C$ 仍共享，与"视图特有"的口号存在张力。

**实验数字（AP ↑）**：3Sources **0.4676**（次优 MDFS 0.4481）、SCENE **0.7953**（MRDM 0.7929）、
IAPRTC12 **0.1436**（MRDM 0.1400）、Corel5K **0.2361**（M2LD 0.2355）、ESPGame **0.2248**（CLML 0.2238）、
OBJECT **0.4873**（MDFS 0.4756）。唯一被反超的是 3Sources 的 HL（0.2337 vs ELSMML 0.2331）。
**优势幅度同样很小**（SCENE 仅 $+0.0024$）。

#### 3.6.5 代码对照：公式 ↔ `alg/I2VSLC.py`

| 论文公式 | 代码位置 | 对应关系 |
|---|---|---|
| 式(3) 视图特征图 $S^{(i)}/L^{(i)}$ | L39–49 | `construct_W(v[i], knn k=5, heat_kernel, t=1.0)` —— **建在特征块上**（与论文一致） |
| 式(19)(20) $w^{(i)}$ | L67–76 | 两个对角阵：`D = diag(0.5/‖w[i]‖_F)`（视图级）、`C = diag(0.5/‖w_row‖₂)`（特征级） |
| 式(24) $C$ | L80–81 | `AA ∘ (Yᵀ·sum_Yt + YᵀY)/(2YᵀY·AA)`（KKT 正确版） |
| 式(27) $y^{(i)}$ | L85–95 | `tem1`/`tem2` 被二值化 ⇒ **$F$ 与 $G$ 退化成同一个掩码**（§5.2 H 行） |
| 式(14) 目标值 | L106–157 | `temp1..temp5`；inter-view 项只累加 $j>i$（全对的一半），而更新式用 $(V{-}1)$（全对） |
| 硬二值化 | L96–103 | 每轮对 $y^{(i)}$ 做 MinMax + 0.5（论文无此步） |

```python
# L64-76  w⁽ⁱ⁾ 的更新：两个对角阵分别对应 γ（视图级）与 δ（特征级）
for i in range(n_view):
    d_temp = np.full((m[i], 1), LA.norm(w[i], 'fro'))
    D = np.diag((0.5 / d_temp).flat)                 # 视图级：一个标量，整体缩放 w⁽ⁱ⁾
    c_temp = np.sqrt(np.sum(np.multiply(w[i], w[i]), 1) + eps)
    C = np.diag((0.5 / c_temp).flat)                 # 特征级：逐行重加权（行 2-范数）
    w[i] = np.multiply(w[i], np.true_divide(
        np.dot(v[i].T, Y),                           # ← 论文式(19) 此处应是 y⁽ⁱ⁾（§5.2 M 行）
        np.dot(v[i].T, np.dot(v[i], w[i]))
        + gamma * np.dot(D, w[i]) + lamb * np.dot(C, w[i]) + eps))
```

```python
# L80-81  标签相关矩阵 C（代码里叫 AA，初始化为单位阵）
AA = np.multiply(AA, np.true_divide(np.dot(Y.T, sum_Yt) + np.dot(Y.T, Y),
                                    2 * np.dot(np.dot(Y.T, Y), AA) + eps))
```

```python
# L84-95  y⁽ⁱ⁾ 的更新：intra（Sx/Ax）+ inter（tem1/tem2）
tem1 = sum_Yt - y[i]; tem1[tem1 == 0] = 0; tem1[tem1 != 0] = 1   # 二值化后的指示器
tem2 = y[i] - sum_Yt; tem2[tem2 == 0] = 0; tem2[tem2 != 0] = 1   # ← 与 tem1 完全相同
y[i] = np.multiply(y[i], np.true_divide(
    alpha * np.dot(Sx_lst[i], y[i]) + alpha * (n_view - 1) * tem1
    + np.dot(v[i], w[i]) + beta * (np.dot(Y, AA) + tem2),
    alpha * np.dot(Ax_lst[i], y[i]) + alpha * (n_view - 1) * y[i]
    + y[i] + beta * y[i] + eps))
```

$F=Y_{vs}-y^{(i)}$ 与 $G=y^{(i)}-Y_{vs}$ 本来是一对**有符号的连续差**，但代码把两者都二值化成
同一个 0/1 掩码（`tem1 ≡ tem2`），视图间共识项因此**退化为"在不一致的位置统一加一个常数拉力"**，
丢掉了差的幅值信息 —— 实测两种口径给出的 $\Phi$ 相差 **342%**（§5.2 H 行）。

另外两处：每轮对 $y^{(i)}$ 做 MinMax + 0.5 硬二值化（论文只对 $Y_{vs}$ 用 $\sum\ge1$ 的绝对阈值）；
目标函数里 inter-view 项只算 $j>i$（全对的一半），而更新式用 $(V-1)$（按全对计）⇒
**记录的 $\alpha$ 与驱动的 $\alpha$ 差 2 倍**（§5.2 M 行）。

---

## 4. 论文之间的"问题→改进"闭环

这是本文档最重要的一节：这六篇不是并列的，而是**前一篇的软肋构成后一篇的动机**。

### 4.1 完整演进图

```
┌─────────────────────────────────────────────────────────────────────────┐
│  轴 I：标签拓扑                                                          │
│                                                                         │
│  共享 Y ──► y⁽ⁱ⁾ 按视图拆 ──► Y 拆三类 + 逻辑约束 ──► 加"分层"形式定义   │
│             (I²VSLC)          (DHLI)                 (TOCL)             │
│                │                  │                      │              │
│                │                  └── 遗留：噪声靠 ℓ1 稀疏，            │
│                │                      未建模"错误标签的模式"            │
│                └── 遗留：C 仍全局共享（与"视图特有"口号有张力）          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  轴 II：视图拓扑                                                         │
│                                                                         │
│  直接拼接 ──► 固定权重 ΣvᵢX⁽ⁱ⁾ ──► X^f 成为待解变量 ──► vᵢ 随 G 演化     │
│                (UGRFS/GRAFS 形式)   (GRAFS)            (UGRFS)          │
│                                        │                   │            │
│                                        │                   └── 遗留：    │
│                                        │                      vᵢ 公式尺度 │
│                                        │                      不敏感(§5)  │
│                                        └── 遗留：X^f 与标签仍是单层关系   │
│                                                                         │
│  ──► 融合塞进损失（EF²FS）──► 张量约束跨视图高阶关系（TOCL）            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  轴 III：损失拓扑（"谁拟合谁"）                                          │
│                                                                         │
│  特征→标签(单层) ──► 重构视图→标签(两层) ──► 三方共同拟合 G(多边)        │
│  (DHLI/I²VSLC)        (GRAFS)                (EF²FS)                    │
│                                                    │                    │
│  ──► 双向互推 D ⇄ X⁽ⁱ⁾ (UGRFS) ──► 线性+非线性并行"对立互补" (TOCL)    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  轴 IV：优化拓扑                                                         │
│                                                                         │
│  乘性更新（六篇全部）──► ALM + ADMM（THBFS，最新工作）                   │
│         │                                                                │
│         └── 遗留：无收敛定理；实测震荡（本文档 §5.7 首次量化）           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 时间线上的"动机继承"关系（带证据）

| 后篇 | 前篇的软肋（原文证据） | 后篇的改进 |
|---|---|---|
| **I²VSLC** (2024) | 前人"assume an ideal uniform label matrix" | 引入 $y^{(i)}$ + $\Phi(\cdot)$ 双层标签关系 |
| **GRAFS** (2024) | Zhu et al. 的全局视图"synthesized by **concatenating** the original features" | $X^f$ 变成待解潜变量 + 锚点双路径 |
| **EF²FS** (2025) | Zhu et al. 是"**two-step** method"；综合部分"relatively limited, often relying on simple techniques such as **direct concatenation**" | 融合进入损失函数，$c_i$ 与 $G$ 闭环 |
| **UGRFS** (2025) | ①"extract information **separately**… **unclear segmentation**"；②"the reliability of each view, and even each sample, is uniformly equal to **1**" | ①用一个全局视图变量取代分割；②引入 $C^{(i)}$ |
| **TOCL** (2025) | "prior studies often **overlook different types of the mapping** and **higher-order constraints** among view-specific labels" | 双路映射（线性+非线性）+ 张量核范数 |

**这张表本身就是这个研究组的方法论**：每篇的 Introduction 里都有一段
"前人做了什么 → 但忽略了什么"，而那个"忽略"恰好是下一篇要做的。
**这是高产的写作模板，也是理解其创新点的钥匙。**

### 4.3 两两之间的技术血缘

```
                    ┌──────────────────────────────┐
                    │  DHLI (AAAI'24)              │
                    │  • hybrid labels (Y_c/Y_s/Y_n)│
                    │  • 逻辑算子 ⊗⊕⊖∘              │
                    │  • 视图互斥 F=⊙_{j≠i}y⁽ʲ⁾     │
                    └───────────┬──────────────────┘
                                │  继承：标签三类 + ⊖ 算子 +
                                │        留一 Hadamard 互斥
                                ▼
                    ┌──────────────────────────────┐
                    │  TOCL (ACM MM'25)            │
                    │  • Def.3/4 分层标签相关       │
                    │  • P2 = ‖Y_all⊗Y_n⊖Y‖²+‖Y_n‖² │
                    │  • 新增：张量核范数 + 加权 TNN│
                    │  • 新增：双路映射对立互补      │
                    └──────────────────────────────┘

   ┌────────────────────┐        ┌────────────────────┐
   │ I²VSLC (InfSci'24) │        │ GRAFS (InfSci'24)  │
   │ • y⁽ⁱ⁾ + Φ(·)      │        │ • X^f 潜变量        │
   │ • 每视图一张 L⁽ⁱ⁾   │        │ • 锚点双路径        │
   └─────────┬──────────┘        └─────────┬──────────┘
             │  把"重构视图"作为拟合目标    │
             └──────────────┬──────────────┘
                            ▼
              ┌──────────────────────────────┐
              │  UGRFS (AAAI'25)             │
              │  • D = Y_x W_y（全局视图分布）│
              │  • C⁽ⁱ⁾ 样本置信度（新增）    │
              │  • vᵢ ∝ 1/Tr(GᵀL⁽ⁱ⁾G)（随G演化）│
              └──────────────────────────────┘
                            │
                            │  把"融合"塞进损失、让 c_i 与 G 闭环
                            ▼
              ┌──────────────────────────────┐
              │  EF²FS (PR'25)               │
              │  • G 一物两用（去噪器+嵌入） │
              │  • Σ‖a⁽ⁱ⁾−w⁽ⁱ⁾‖² 双层一致    │
              └──────────────────────────────┘
```

**注意 UGRFS 的位置**：它同时吸收了两条线 —— 从 GRAFS 拿走"$X^f$ 是潜变量"，
从 EF²FS 拿走"视图权重随嵌入演化"，再加上自己的 $C^{(i)}$。这就是它能上 AAAI 的结构性原因。

---

## 5. 代码与论文的偏差总表（含新发现的缺陷）

### 5.1 偏差分类

| 级别 | 含义 | 影响 |
|---|---|---|
| **H** | 改变模型语义/容量 ⇒ 影响结论 | 必须报告 |
| **M** | 影响迭代轨迹/可比性 ⇒ 影响复现 | 复现时必须记录 |
| **L** | 数值加固/命名 ⇒ 无实质影响 | 可忽略 |
| **F** | 代码修正了论文的笔误 | 代码对、论文错 |

### 5.2 逐篇偏差

#### TOCL
| 级别 | 偏差 | 位置 |
|---|---|---|
| **H** | **t-SVD 的"管方向"错了**：论文 Definition 2 / Eq.(10) 把 $T$ 定义为 $n\times V\times n$（第 2 阶是**视图**），故 FFT 应沿视图轴（长度 $V$）；但代码构造 `HH2 = HH.transpose((0,2,1))` 形状 $(n,V,n)$，而算子内部对**第 2 阶（长度 $n$）** 做 FFT ⇒ **在样本方向而非视图方向做 t-SVD** | `alg/TOCL.py` L20, L38–63 |
| **H** | **偶数个视图时 IndexError**：`n3 % 2 == 0` 分支用 `i = halfn3 + 1`，当 `n3=2` 时 `i=2` 越界。实测 $V=2$ 直接崩溃（本文档 `insight/mechanisms.py` 已修复） | L52–54 |
| **M** | **从不处理 $j=0$ 切片** ⇒ `X[:,:,0]` 恒为零矩阵，重构丢掉直流分量 | L25–63 |
| **M** | 循环体内先算 $i$、再写 `X[:,:,n3-i] = conj(X[:,:,i])`，使 $i\ge2$ 的**计算结果被镜像覆盖**，但 `wtnn` 的累加发生在覆盖**之前** ⇒ **累加到的切片数与实际写入张量的切片集合不一致** | L49, L62 |
| **L** | 目标函数 `2*beta*temp6`、`2*lamb*temp9` 的因子 2 与论文式(14) 的 $\beta\|P\|_F^2$ 不一致 | L247–248 |
| **L** | `delta`、`aaa`、`rho` 在函数内写死为 1，论文未说明 | L77–79 |
| **L** | 收敛阈值 $10^{-6}$（其他算法 $10^{-3}$） | L261 |

#### UGRFS
| 级别 | 偏差 | 位置 |
|---|---|---|
| **H** | **视图权重 Eq.(4) 被换成另一个量（角色互换）**：论文 $v_i\propto1/\mathrm{Tr}(y^\top L_x^{(i)}y)$（标签 × **视图特征图**），代码 $1/\mathrm{Tr}(X^{(i)\top}L_yX^{(i)})$（**视图特征** × 标签图）。且代码**从不构造任何视图特征图** | L74–79 vs L66–71 |
| **H** | **Eq.(6) 的 $\rho(Y)$ 被简化成标准高斯核，且带宽方向相反**：论文 $\rho(Y)=\exp(-[\cdots]/\mathrm{avg}(p\,dist)^2)$，代码 $\exp(-\|y_i-y_j\|^2/(2\sigma^2))$ with $\sigma=1/\mathrm{avg}(p\,dist)$。分子里的 $J_1Y^\top(\sum y^2)-YY^\top$ **完全没实现**，且论文 $\sigma=\mathrm{avg}$、代码 $\sigma=1/\mathrm{avg}$，**两者不可能互相化简** | L86–87 |
| **M** | **$\ell_{2,1}$ 项的因子 2 丢失**：论文式(14) 分母是 $2\delta EW^{(i)}$，代码 `lamb*np.dot(D,w[i])` 中 `D=diag(0.5/‖W_i‖)` ⇒ **有效稀疏权重只有论文的一半**（要与论文一致需 `lamb` 取 $2\delta$） | L106–108, 112 |
| **M** | 论文的 $E$ 在代码里叫 `D`，而**论文的全局视图分布 $D$ 在代码里无同名变量**（`p2 = Yx@Wy`）⇒ 极易误读 | L108 vs L126 |
| **M** | $C^{(i)}$ 更新复用**旧的** `p1 = c_old@x[i]` 作为分母里的 $A^{(i)}$，却配刚更新的 $w[i]$（Jacobi 式混合，论文未声明） | L110–114 |
| **L** | `objectives/2` 使首轮 cver ≈1 —— 正对应论文 Fig.5 的 "we set the initial point to approximately one"，但论文正文完全没解释 | L140–141 |
| **L** | 停机阈值论文只说 "such as 0.001"，未给迭代上限；代码 `cver<1e-3`、上限 500 | L149, L100 |
| **L** | **论文无任何实测耗时/内存数字**（只有复杂度分析），这是复现对比的空档 | 论文自身问题 |
| **L** | 论文 Eq.(11) 的 $\|W\|_{2,1}$ 无系数、Eq.(12) 写作 $\delta\|W\|_{2,1}$；Eq.(7) 的 $D=\rho(Y)\hat W_y+b$ 量纲不自洽 | 论文自身问题 |

#### EF²FS
| 级别 | 偏差 | 位置 |
|---|---|---|
| **H** | **$A$ 的更新用未加权的原始拼接矩阵 `X`**，而论文式(25) 用加权融合矩阵 $X^f$；同一份代码在 $G$ 更新与目标函数里都用 `new_X` ⇒ **$A$ 的更新方向不是所记录目标函数的下降方向**（仅当所有 `nu[i]=1` 时一致） | L92–94 vs L87, L105 |
| **M** | 论文 Table 1/§3.1 把 $A$ 写成 $d\times c$、$w^{(i)}$ 写成 $d^{(i)}\times c$，与式(13)(14)(17) 要求的 $k$ 维矛盾（从 $Y$ 版改到 $G$ 版时未同步改维度） | 论文自身缺陷 |
| **M** | 视图权重步在 $G$ 更新**之前**执行，用的是**上一轮的 $G$**（滞后一轮），而式(12) 定义在当前 $G$ 上 | L62–77 |
| **M** | **论文全文未给隐空间维数 $k$ 的取值/选取规则**，而代码必须传 `V_dim`（`main.py` 默认 30） | — |
| **M** | 论文 Table 3/4 与 Table 7 对同一个 EF²FS 给出不同数值（MIRFlickr AP 0.6759 vs 0.6518；IAPRTC12 AP 0.1509 vs 0.1418），论文未说明 | 论文自身问题 |
| **L** | 停机 `cver<1e-3`，上限 `max_iter(100)`（其他算法 500） | L129, L59 |
| **L** | 停机用 `abs()` 而论文 Fig.4 定义的是**带符号**的比值 | L125 |
| **L** | §4.2.4 出现 "The complexity of CLML is comparable to that of **GRAFS**"，而 GRAFS 并非本文基线 —— 疑为复制粘贴残留 | 论文自身问题 |

#### DHLI
| 级别 | 偏差 | 位置 |
|---|---|---|
| **H** | **每轮对 `y[i]`、`Y_c`、`Y_n` 做 MinMax(逐列)+0.5 硬二值化**，论文完全没有这一步。后果：①变量强制为 {0,1}、每列必有一个正例；②0 是乘性更新的**吸收态**；③乘性更新之间插入硬投影，**破坏论文所声称的收敛性前提** | `alg/DHLI.py` L98–103, 120–124, 136–140 |
| **H** | $Y_{spe}$ 语义前后不一致：初始化把负差归 0（$\mathbf 1[Y>Y_c+Y_n]$），循环内把负差归 1（$\mathbf 1[Y\ne Y_c+Y_n]$）。后者才与 $\ominus$ 语义一致 | L55–57 vs L173–176 |
| **M** | $Q=1/(2\lvert Y_n\rvert)$ 在二值下使 $\beta QY_n$ 退化为常数 $\beta/2$ ⇒ 式(17) 的"自适应 $\ell_1$ 重加权"**名存实亡** | L131–132 |
| **M** | $A=\sum_iX^{(i)}W^{(i)}$ 在视图循环内累加 ⇒ 一次迭代中混用已更新/未更新的 $W^{(i)}$ | L95 |
| **M** | 排序前对 $W$、$U$ 各自 MinMax 再相加；论文 Algorithm 1 只写 $\|(W+U)_{(j)}\|_2$，无归一化 | L202–212 |
| **M** | 目标函数里 β 项是 $2\beta\|Y_n\|_1$（论文是 $\beta\|Y_n\|_1$）⇒ 只影响停止判据 | L154 |
| **L** | 论文的 $\delta$ 在代码里叫 `lamb` | L13, 219 |
| **L** | 初始化全随机（$w,u\sim U[0,1)$、$y,Y_c,Y_n$ 随机二值），论文只写 "Initialize" | L23–27, 45–46 |
| **L** | 分母统一 `+eps=2.2204e-16` | L10 等 |
| — | **已修正的上游 bug**：互斥乘积原写成 `B=0; B=B*y[i]`（恒为 0 且含自身），使互斥项静默失效。现为 `B=∏_{j≠i}y[j]` | L80–87 |

#### GRAFS
| 级别 | 偏差 | 位置 |
|---|---|---|
| **H** | 论文式(2)(3)(5) 的**视图**亲和图 $S^{(v)}/L_x^{(v)}$ **完全未实现**；代码改为在**标签矩阵**上建一个 knn(k=20) 热核图，$v_i\propto1/\mathrm{Tr}(X^{(i)\top}L_yX^{(i)})$。**语义转置** | L82–96 |
| **H** | $C^{(i)}$ 被强制为对角阵（论文是满矩阵）⇒ 参数从 $O(d^{(i)2})$ 降到 $O(d^{(i)})$，无法建模特征间贡献交互 | L60–64, 106–109 |
| **H** | $k_1=10$ 硬编码（论文未给值、未调）；$kk$ 论文未报告（代码默认 20，注释提示曾用 100）⇒ **论文的结构参数不可复现** | L46, L22 |
| **M** | 仍构造 $n\times n$ 稠密标签图（OBJECT $n{=}6047$ 时 `Sy/Ay/Ly` 合计约 0.9 GB），抵消了"用锚点图替代全图"的动机 | L84–89 |
| **M** | 非负性无强制。若数据含负值，$B$ 与 $X^f$ 的分子可能为负，乘性更新会破坏非负假设甚至发散 | L106–138 |
| **M** | $X^f$ 初始化为 $\mathrm{MinMax}(YW^\top)$（标签暖启动，论文未提）；`normalization` 无除零保护 | L52–53, 18–19 |
| **F** | 论文式(27) 分子缺转置（应为 $D^{(i)\top}$）；**代码正确** | L137 |
| **F** | 论文式(17) 的系数写成 $\gamma EW$（应为 $\delta EW$）；**代码正确** | L115 |
| **L** | `argsort(-w_2)` 是**降序**，但 L192 注释写"升序排列，最后一个最重要"——**注释错、实现对** | L174, 192 |

#### I²VSLC
| 级别 | 偏差 | 位置 |
|---|---|---|
| **H** | **$F=Y_{vs}-y^{(i)}$ 与 $G=y^{(i)}-Y_{vs}$ 被二值化成同一个 0/1 掩码**（`tem1 ≡ tem2`）。论文的 $F,G$ 是有符号连续差 ⇒ 视图间共识项**退化为"对不一致位置统一加常数拉力"**，丢失梯度幅值信息。实测二者给出的 $\Phi$ 相差 **342%** | L85–92 |
| **H** | **每轮对 $y^{(i)}$ 做 MinMax+0.5 硬二值化**，论文无此步骤（论文只对 $Y_{vs}$ 用 $\sum\ge1$ 绝对阈值） | L96–103 |
| **M** | $C$ 更新用 KKT 正确版 $\frac{Y^\top Y_{vs}+Y^\top Y}{2Y^\top YC}$，与论文印刷的式(24) 不同（论文式(23) 的 $Y^\top Y$ 符号写反）⇒ 定点语义相同但迭代轨迹不同 | L80–81 |
| **M** | $w^{(i)}$ 更新分子用观测 $Y$ 而非 $y^{(i)}$ —— **论文自身的笔误被代码沿用**（式(19)(20)(21) 用 $Y$，而式(14)(18) 的拟合项是 $y^{(i)}$） | L75 |
| **M** | 目标函数视为 $i<j$（是全对的一半），而更新式用 $(V-1)$（全对计）⇒ 记录的 $\alpha$ 与驱动的 $\alpha$ 差 2 倍 | L115–116 vs L94 |
| **M** | β 项用二值化后的 $YC$，第二子项是 $\|\mathbf 1[Y\ne\hat{YC}]\|^2$ 而非 $\|Y-YC\|_F^2$ | L122–126, 143–148 |
| **L** | 辅助阵命名与论文互换（代码 `D`=论文 $E^{(i)}$、代码 `C`=论文 $D$）；但代码的**分工是数学上正确的**（$\gamma$ 项用标量×I 松弛、$\delta$ 项用逐行重加权） | L67–73 |
| **L** | 停机 `cver<1e-3` 且 `iter>2`，上限 500；论文只写 "until Convergence" | L167 |
| **L** | `temp2` 为 NaN 时打印并 break（论文无） | L117–119 |

> **⚠️ 文档勘误**：`PAPERS_AND_CODE.md` §5 与 `pre-pdf/paper-notes/I2VSLC.md` 写作
> "$L^{(i)}$ 是**标签图**拉普拉斯""在**标签空间**做流形平滑"。
> 论文式 (3) 的距离明确取 $\|x_i^{(v)}-x_j^{(v)}\|$，代码 L44 也是在**特征块**上建图。
> 正确表述是：**图建在视图特征空间，正则项作用在标签矩阵上**。

### 5.3 六个"论文自身"的问题（与代码无关，但影响可复现性）

| 论文 | 问题 |
|---|---|
| TOCL | 加权 TNN 声称的权重 $w_i=C/(\sigma_i+\varepsilon)$ 与其给出的闭式解（Eq.24）**不自洽**（§5.6 有实证） |
| UGRFS | 无实测耗时/内存；$\rho(Y)$ 与 $D=\rho(Y)\hat W_y+b$ 两处公式本身有问题 |
| EF²FS | 隐空间维数 $k$ 未给取值；Table 3/4 与 Table 7 数值不一致 |
| DHLI | 贡献点声称 "with **proven** convergence"，但 9 页正文无任何定理/引理/附录；同页收敛段误写成 "the convergence of **LGCM**"（他文残留） |
| GRAFS | 结构参数 $k,k_1$ 未做敏感性分析、未报告取值 |
| I²VSLC | **完全没有消融实验**（全文无 "ablation"） |

### 5.4 一个跨篇的共同缺陷：视图权重公式（新发现）

**这是本文档最重要的实证发现**，影响 UGRFS / GRAFS / EF²FS / I²VSLC 四篇。

四篇都用同一个视图权重形式：

$$
v_i=\frac{1/\mathrm{Tr}\big(G^\top L^{(i)}G\big)}{\sum_j1/\mathrm{Tr}\big(G^\top L^{(j)}G\big)}
\qquad\text{或}\qquad
v_i\propto\frac{1}{\mathrm{Tr}\big(X^{(i)\top}L_yX^{(i)}\big)}
\tag{5.1}
$$

**它有两个结构性缺陷**（`python -m insight.run_demo M3 M5 M8` 可复现）：

**(1) 对 $G$ 的尺度是二次的**：

$$
\mathrm{Tr}\big((cG)^\top L(cG)\big)=c^2\,\mathrm{Tr}\big(G^\top LG\big)
\tag{5.2}
$$

实测 $c=0.1,1,10$ 时能量为 $0.0116,\ 1.164,\ 116.4$ —— 严格按 $c^2$ 缩放。
因此**特征取值小的视图被系统性高估**，与它是否携带标签信息无关。

**(2) 与标签完全无关**：$L^{(i)}$ 来自特征，$G$ 来自特征，**标签 $Y$ 不出现在分子分母任何一处**。
后果是一个纯噪声视图只要几何"紧"就能拿高权重。

**(3) 热核下溢导致权重塌缩**（最严重的后果）。实测构造
（三视图：真信号 / 信号+噪声 / 纯噪声，$n=200$）：

| 视图 | $\|L^{(i)}\|_F$ | 论文原式 $E$ | $v_i$(论文) | 标签感知 $E$（修正） | $v_i$(修正) |
|---|---|---|---|---|---|
| 真信号 | 0.810 | 1.1717 | **0.4191** | 25.71 | 0.3359 |
| 信号+噪声 | **0.000000** | $\approx 0$ | **0.0000** | 15.84 | 0.5453 |
| 纯噪声 | 0.073 | 0.8453 | **0.5809** | 72.69 | 0.1188 |

**论文原式把最高权重（0.581）给了「纯噪声」视图，而把「信号+噪声」视图压到 0**
—— 多视图融合名存实亡。根因：该视图的中位平方距离远大于 $t=1.0$，
$\exp(-d^2/(2t^2))$ 在 float64 下下溢为 0，$S$ 退化成只有自环的单位阵、
$L\equiv0$、$\mathrm{Tr}(G^\top LG)=0$，于是它的 $1/E$ 被 `1/eps` 兜底成极大值。

> **关于数值的说明**：论文原式具体把最高权重给哪个视图，取决于随机初始化的
> $G$；**稳定可复现的是"权重塌缩"这一现象本身**（总有视图被压到 0）。
> 修正版则稳定地把 0.88 的总权重分给两个含信号视图、0.12 给纯噪声视图。

**修正方案**（`insight/mechanisms.py::_label_aware_view_energy`，已实现并验证）：

$$
E_i=\frac{1}{d^{(i)}}\min_{W}\Big\|\tilde S^{(i)}Y-X^{(i)}W\Big\|_F^2+\lambda\|W\|_F^2,
\qquad
\tilde S=D^{-1/2}SD^{-1/2}
\tag{5.3}
$$

性质：①**尺度不变**（度归一化把行和钉成常数）；②**维度可比**（除以 $d^{(i)}$ 消除
自由度偏差）；③**标签相关**（$\tilde SY$ 是"按该视图邻域平滑后的标签"，残差小
意味着"该视图的局部几何能解释标签"—— 这才是"视图重要"的定义）；④**有闭式解**
$W=(X^\top X+\lambda I)^{-1}X^\top\tilde SY$。

**修正后权重变为 $(0.336,\ 0.545,\ 0.119)$** —— 两个含信号视图合计 0.88，
纯噪声视图 0.12，**符合直觉**（而论文原式是 $(0.419,\ 0.000,\ 0.581)$）。

### 5.5 另一个跨篇缺陷：$\gamma$ / $t$ 的量纲问题（新发现）

同一类问题的另一种表现。在自适应图学习（`insight/mechanisms.py::adaptive_graph_learning`）中，
$\gamma$ 乘在 $\|f_i-f_j\|^2$ 上（其量纲是 $O(c/n)$），而 $d_{ij}=\|x_i-x_j\|^2$
的量纲是**特征尺度的平方**。二者必须可比，否则：

* $\gamma$ **过小** → 距离项主导 → $S$ 退化为 1-NN 图，连通分量数 $\approx n$（实测 240/240）；
* $\gamma$ **过大** → $F$ 项主导 → $S$ 趋于均匀图。

**只有 $\gamma\approx\mathrm{median}(d^2)/(2c/n)$ 附近才是正确区间**。
好消息是自动标定后它在 $[10^{-3},10^{6}]\times\gamma^*$ 的**9 个数量级**内都正确工作
（实测均恢复恰好 4 个连通分量），而固定 knn 图对 $t$ 的容差要窄得多
（$t=0.1$ 时下溢、$t=30$ 时近全连接，$\|L\|_F$ 跨 7556 倍）。

**但自适应图并非"免超参"**：它是把 2 个超参 $(k,t)$ 换成 1 个**有量纲、可自动标定**的 $\gamma$，
并把结构约束变成可验证的 $\mathrm{rank}(L_S)=n-c$。这个诚实的表述比"免超参"更站得住脚。

### 5.6 TOCL 加权 TNN 的内部不自洽（新发现）

论文声称权重 $w_i^{(j)}=C/(\sigma_i^{(j)}+\varepsilon)$（大奇异值少惩罚），
并给出闭式解 $\sigma^*=(c_1+\sqrt{c_2})/2$。但这两者**并不等价**：

$$
\underbrace{\sigma^*=\frac{c_1+\sqrt{c_2}}{2}}_{\text{论文 Eq.24（在 }c_2<0\text{ 处硬截断为 0）}}
\qquad\text{vs}\qquad
\underbrace{\sigma^*=\max\!\Big(\sigma-\frac{C}{\sigma+\varepsilon},0\Big)}_{\text{由 } w_i=C/(\sigma_i+\varepsilon)\text{ 严格推出的近端解}}
\tag{5.4}
$$

实测（$C=1$）：

| $\sigma$ | 0.2 | 1.0 | 3.0 | 10.0 | 50.0 |
|---|---|---|---|---|---|
| 论文 Eq.24 | 0.0000 | 0.0000 | 2.618 | 9.899 | 49.98 |
| 严格近端解 | 0.0000 | $\sim0$ | 2.6667 | 9.900 | 49.98 |

两者数值接近但**在 $\sigma\approx C$ 邻域行为不同**：Eq.24 在判别式 $c_2<0$ 处
**硬截断为 0**（不连续），而 $C/(\sigma+\varepsilon)$ 的近端解是**连续软阈值**。
即：论文"声称的加权范数"与其"给出的闭式解"不是同一个东西 —— 更可能是先有
MATLAB 代码（`prox_wtnn`）后有论文的"推导"。

### 5.7 最重要的一条批评：目标值曲线不可作为收敛证据（新发现）

六篇统一画 $y=(z^{t-1}-z^t)/z^{t-1}$。**这个量对震荡序列反而更大**：
相邻两轮差异越大、比值越大，曲线看起来"下降越快"。

实测（`python -m insight.run_demo M3`）：EF²FS 的完整交替迭代在 400 轮内
目标函数最大值/最小值之比达 $7.5\times10^{8}$，末轮值是历史最优值的 **38.5 倍**；
对视图做列标准化后震荡比仍为 $1.5\times10^{12}$、末轮是最优值的 **153.5 倍**。
也就是说：

* 迭代**完全没有收敛**（非单调、大幅震荡）；
* 但按论文那套 $y$ 轴，这条曲线依然是"快速下降后稳定"的形状。

**根因**：乘性更新 $Z\leftarrow Z\circ[\nabla\Theta]^-/[\nabla\Theta]^+$ 只做非负缩放，
**没有任何步长控制**；当 $X^\top X$ 病态或尺度失衡时，$Z$ 会在不同尺度间来回跳。
数据标准化能改善数值条件，但**不能消除**震荡（实测标准化后震荡比仍达 $1.5\times10^{12}$）。

**这正是最新工作 THBFS 改用 ALM+ADMM 的原因**，也是 §6.7 的改进方向。

---

## 6. 八条可验证的改进思路：公式 + 代码

每条给出：**动机（来自 §5 的哪个缺陷）→ 公式 → 代码 → 验证方式**。
代码全部在 `insight/mechanisms.py`，验证在 `insight/run_demo.py`，
运行方式 `python -m insight.run_demo M1 ... M8`（已全部通过）。

### 6.1 M1 — 修正后的加权张量核范数近端算子（TOCL 的 D1/D2/D3/M1/M2）

**动机**：§5 TOCL 的管方向错误（H）、偶数视图崩溃（H）、$j{=}0$ 切片缺失（M）、
$wtnn$ 累加与写入不一致（M）。

**公式**：严格近端解

$$
\sigma_i^\star=\max\!\Big(\sigma_i-\frac{C}{\sigma_i+\varepsilon},\,0\Big)
\tag{6.1}
$$

**代码**（`insight/mechanisms.py`）：

```python
def prox_weighted_tnn(T, C, fft_axis=2):
    """加权 TNN 近端算子；fft_axis 显式指定「管方向」。
    论文语义：T ∈ R^{n×V×n} 第 2 阶是视图 ⇒ fft_axis=1。
    参考实现：HH2 = HH.transpose((0,2,1)) 后对第 2 阶 FFT ⇒ fft_axis=2。
    """
    That = np.moveaxis(_fft(np.moveaxis(T, fft_axis, -1), axis=-1), -1, fft_axis)
    loop_len = That.shape[fft_axis]
    Tnew_hat = np.zeros_like(T, dtype=complex)
    for i in range(1, round(loop_len / 2)):          # ← 复刻参考实现的循环（含其缺陷）
        U, Sv, Vh = LA.svd(_slice_along(That, fft_axis, i), full_matrices=False)
        V = Vh.conj().T
        c1, c2 = Sv - EPS, (Sv - EPS) ** 2 - 4.0 * (C - EPS * Sv)
        keep = c2 > 0.0
        Snew = np.zeros_like(Sv)
        Snew[keep] = np.maximum(c1[keep] + np.sqrt(c2[keep]), 0.0) / 2.0
        ...
    if loop_len % 2 == 0:                            # ← 修好偶数视图的越界
        i = round(loop_len / 2) + 1
        if i < loop_len: ...
    Tnew = np.real(np.fft.ifftn(np.moveaxis(Tnew_hat, fft_axis, -1), axes=[-1]))
    return np.moveaxis(Tnew, -1, fft_axis), wtnn / loop_len, trank
```

**验证**（M1）：① $C$ 增大 → tubal rank 单调不增 $(68,14,0,0)$；
② 与 `alg/TOCL.py` 的**逐切片秩结构完全一致**；③ 修复了 $V=2$ 的崩溃；
④ 用 `fft_axis=1` 才得到论文语义（tubal_rank 14 vs 参考实现的 3）。

---

### 6.2 M2 — 样本可信度的闭式含义（UGRFS 的 C 更新）

**动机**：UGRFS 的 $C^{(i)}$ 更新式在论文里是"多变量混在一起"的矩阵式，
看不出 $c$ 到底在度量什么，也无法判断它的效应量。

**公式**：把 $C$ 取对角后，逐样本的不动点是

$$
c_m^\star=\frac{\big[X(YW^\top)\big]_{mm}}{\big[X(WW^\top)X^\top\big]_{mm}}
=\frac{\text{该样本的特征-标签拟合质量}}{\text{该样本投影后的能量}}
\tag{6.2}
$$

**这条化简本身就是一个贡献**：它把"样本不确定度"从超参变成了**有闭式含义的残差解释比**，
并说明 $\beta$ 项在分子分母上**完全抵消**（$\beta$ 不影响不动点，只影响收敛速度）。

**代码**：

```python
def uncertainty_confidence_update(X, Y, W, D, c, beta):
    num = np.diag(X @ W @ Y.T) + beta * np.diag(X @ X.T)
    den = np.diag(X @ W @ W.T @ X.T) + beta * np.diag(X @ X.T) + EPS
    return np.clip(c * (num / den), 1e-6, None)
```

**验证**（M2）：注入"特征尺度与干净样本相当、但标签随机"的噪声样本后，
$c_{\text{noisy}}/c_{\text{clean}}=0.938$，以中位数为阈值可识别 **57.5%** 的噪声
（基线 50%）。

**⚠️ 诚实报告效应量**：这是一个**弱效应**（相对分离度仅 6.24%）。
这恰好解释了为什么 UGRFS 消融里 v1（去样本置信度）只掉 1.8%，
也说明论文只给出"置信度可视化"（证明其**非均匀**）而没有证明它能**准确识别噪声**
—— 本实验支持同样的保守结论。

---

### 6.3 M3 — 嵌入式融合 + 标签感知视图权重（EF²FS 的 D1 + §5.4）

**动机**：§5.4 的视图权重缺陷（尺度二次敏感 + 与标签无关 + 热核下溢导致权重塌缩）。

**公式**：

$$
c_i=\frac{1/E_i}{\sum_j1/E_j},\qquad
E_i=\frac{1}{d^{(i)}}\min_{W}\big\|\tilde S^{(i)}Y-X^{(i)}W\big\|_F^2+\lambda\|W\|_F^2,
\quad \tilde S=D^{-1/2}SD^{-1/2}
\tag{6.3}
$$

**代码**：

```python
def ef2fs_step(views, Y, G, B, A, ws, Lx_lst, alpha=1., beta=1., gamma=1.,
               view_weight="paper", Y_for_weight=None):
    if view_weight == "paper":                 # 论文原式（尺度不敏感、与标签无关）
        energy = [np.trace(G.T @ Lx_lst[i] @ G) for i in range(V)]
    else:                                      # 修正版：标签感知 + 尺度不变
        energy = [_label_aware_view_energy(views[i], Y_for_weight, Lx_lst[i])
                  for i in range(V)]
    energy = np.where(np.abs(energy) < 1e-10, 0.0, energy)   # 浮点负值截断
    inv = np.where(energy > 0, 1/np.where(energy > 0, energy, 1), 1/EPS)
    nu = inv / inv.sum()
    ...
```

**验证**（M3）：
* 论文原式 → $(0.000,\ \mathbf{1.000},\ 0.000)$（**权重全部塌缩到单一视图**）；
* 修正版 → $(0.336,\ 0.545,\ 0.119)$（含信号视图合计 0.88，纯噪声 0.12）；
* $\mathrm{Tr}(G^\top LG)$ 对 $G$ 的尺度严格按 $c^2$ 缩放（实测 $0.0116/1.164/116.4$）。

---

### 6.4 M4 — 全票通过语义的共识算子（DHLI 的视图互斥项）

**动机**：DHLI 的 $B=\bigcirc_{j\ne i}y_s^{(j)}$ 是一个**离散逻辑量**，
但它出现在连续乘性更新里，其语义（与"软平均"的区别）论文未做量化。

**公式**：增量在"view0 违反共识"与"view0 面对分歧"两种情形下的惩罚对比：

$$
\underbrace{\sum_{j\ne0}\|y^{(0)}-y^{(j)}\|_F^2}_{\text{软平均：A/B 之比 }=2:1}
\qquad\text{vs}\qquad
\underbrace{B=\bigcirc_{j\ne0}y^{(j)}}_{\text{全票通过：A/B 之比 }=1:0}
\tag{6.4}
$$

**代码**：

```python
# 视图互斥矩阵：Hadamard 积的**单位元是 1**（不是 0）
B = np.ones_like(y)
for yj in y_others:
    B = B * yj
y = y * ((X @ u + lamb * tem1) / (y + lamb*y + lamb*y*B*B + EPS))
```

**验证**（M4）：情形 A（其他视图全说 1、view0 说 0）软平均罚 300、$B$ 均值 1.00；
情形 B（其他视图 1:1 分歧）软平均罚 150、$B$ 均值 0.00。
**软平均只给出 2:1 的对比（同量级），$B$ 项给出 1:0（正确区分"违反共识"与"无共识"）。**

> **⚠️ 上游 bug 记录**：原实现写成 `B = 0; B = B * y[i]` ⇒ $B\equiv0$ 且错误地包含自身，
> 使整个互斥项**静默失效**。现版本已修正（`alg/DHLI.py` L84–87）。

---

### 6.5 M5 — 自刻画函数 + 带宽自标定（I²VSLC 的 Φ 与 §5.4）

**动机**：①仓库把论文式(6) 的**有符号连续差**二值化成了同一个 0/1 掩码（§5 I²VSLC H 级偏差）；
②$t=1.0$ 导致热核下溢（§5.4）。

**公式**：

$$
\Phi(\cdot)=\sum_i\mathrm{Tr}\big(y^{(i)\top}L^{(i)}y^{(i)}\big)
+\sum_{i<j}\big\|y^{(i)}-y^{(j)}\big\|_F^2,
\qquad
t_i=\frac{\sqrt{\mathrm{median}(d^2)}}{2}
\tag{6.5}
$$

**代码**：

```python
def self_portrait_objective(ys, Ls, alpha=1.0):
    intra = sum(np.trace(ys[i].T @ Ls[i] @ ys[i]) for i in range(V))
    inter_pairwise = sum(LA.norm(ys[i]-ys[j], 'fro')**2
                         for i in range(V) for j in range(i+1, V))
    # 仓库口径：把有符号差二值化成同一个 0/1 掩码
    union = (sum(ys) > 0).astype(float)
    inter_union = sum(np.count_nonzero((ys[i] != union).astype(float))
                      for i in range(V))
    return {"intra": intra, "inter_pairwise": inter_pairwise,
            "inter_union": inter_union,
            "pairwise": intra + alpha*inter_pairwise,
            "union_indicator": intra + alpha*inter_union}
```

**验证**（M5）：
* 连续伪标签下 $\Phi$(论文软距离) $=772.60$ vs $\Phi$(仓库二值化) $=3418.31$，**相差 342%**；
* 二值情形下两者**数值相等**（$\mathbf 1[y\ne\bar y]$ 计数 $\equiv\sum_{j\ne i}\|y^{(i)}-y^{(j)}\|_F^2$）
  —— 所以这个偏离只在 $y$ **连续**时（即迭代过程中）才咬人，这也解释了为什么代码"看起来能跑"；
* $t=1.0$ 使 view1 的 $\|L\|_F=0.000000$（下溢）；按 $t_i=\sqrt{\mathrm{median}(d^2)}/2$ 标定后
  $t=(3.829,\ 13.279)$，两视图的图差异达 $\|S^{(0)}-S^{(1)}\|_F/\|S^{(0)}\|_F=0.8793$。

---

### 6.6 M6 — 锚点引导的双路径分解（GRAFS）

**动机**：GRAFS 的 $L_E$ 是它最硬的创新，但论文把四个项堆在一个式子里，
"为什么必须共享 $W^c$""为什么 $A^c\approx A^f$ 而不是相等"没有讲透。

**公式**：

$$
L_E=\sum_i v_i\|X^{(i)}-BP^{(i)}\|_F^2
+\|B-W^cA^cR^c\|_F^2
+\|X^f-W^cA^fR^f\|_F^2
+\|A^c-A^f\|_F^2
\tag{6.6}
$$

**代码**：

```python
def grafs_reconstruct(views, B, W1, A1, A2, R1, R2, X_f, nu, alpha=1., beta=1.):
    per_view = []
    for i, Xi in enumerate(views):
        P_i = LA.pinv(B) @ Xi                    # B 固定 ⇒ P⁽ⁱ⁾ 有闭式解
        per_view.append(nu[i] * LA.norm(Xi - B @ P_i, 'fro')**2)
    return {"L_E": sum(per_view)
                   + LA.norm(B - W1 @ A1 @ R1, 'fro')**2
                   + LA.norm(X_f - W1 @ A2 @ R2, 'fro')**2
                   + LA.norm(A1 - A2, 'fro')**2, ...}
```

**验证**（M6）：①$kk$ 从 5 增到 80，重构误差单调下降 $16129\to4511$；
②复杂度压缩比 $O(n^2d)\to O(n\cdot kk\cdot d)$ = **20×**；
③$A^c=A^f$ 时一致性项**精确归零**（$0.0000$）⇒ 该项确实在强制两条分解路径结构相同。

---

### 6.7 M7 — ADMM + 可验证的收敛判据（对六篇"无收敛定理"的补强，§5.7）

**动机**：§5.7 —— 六篇全部无收敛定理，且实测震荡。

**六篇实际求解的问题**（以最简形式）：

$$
\min_{W\ge0}\ \|XW-Y\|_F^2
\qquad\Longrightarrow\qquad
W\leftarrow W\circ\frac{X^\top Y}{X^\top XW}
\tag{6.7}
$$

乘性更新的不动点条件是 $[X^\top(XW-Y)]\circ W=0$，即**非负性下的 KKT 互补松弛** ——
它只是**一阶必要条件**，不保证全局最优，更**不提供 ALM/ADMM 意义下的原始-对偶收敛**。

**ADMM 补强**：引入分裂变量 $Z$ 与对偶变量 $U$：

$$
\begin{aligned}
&\min_{W,Z}\ \tfrac12\|XW-Y\|_F^2+\iota_{\ge0}(Z)\quad\text{s.t.}\ W-Z=0\\
\text{增广拉格朗日}:\quad
&L_\rho=\tfrac12\|XW-Y\|^2+\iota_{\ge0}(Z)+\langle U,W-Z\rangle+\tfrac\rho2\|W-Z\|^2\\
\text{步 1 (W)}:\quad
&W\leftarrow(X^\top X+\rho I)^{-1}\big(X^\top Y+\rho(Z-U)\big)\\
\text{步 2 (Z)}:\quad
&Z\leftarrow\Pi_{\ge0}(W+U)\\
\text{步 3 (U)}:\quad
&U\leftarrow U+\rho(W-Z)
\end{aligned}
\tag{6.8}
$$

**收敛判据**（Boyd et al. 2011, §3.3）：

$$
r_{\text{prim}}=\|W-Z\|_F\to0,\qquad
r_{\text{dual}}=\rho\|Z-Z_{\text{prev}}\|_F\to0
\tag{6.9}
$$

并可按 Boyd §3.4.1 自适应调 $\rho$（$r_{\text{prim}}>10r_{\text{dual}}$ 则 $\rho\leftarrow2\rho$，反之减半）。

**代码**：

```python
def admm_nmf(X, Y, rank, rho=1.0, max_iter=300, tol=1e-6, seed=None):
    W = rng.random((d, l)); Z = W.copy(); U = np.zeros_like(W)
    XtX, XtY = X.T @ X, X.T @ Y
    for it in range(max_iter):
        W = LA.solve(XtX + rho*np.eye(d), XtY + rho*(Z - U))   # 步 1：岭回归闭式解
        Z_prev = Z
        Z = np.maximum(W + U, 0.0)                            # 步 2：非负投影
        U = U + (W - Z)                                       # 步 3：对偶上升
        r_prim, r_dual = LA.norm(W-Z,'fro'), rho*LA.norm(Z-Z_prev,'fro')
        if r_prim > 10*r_dual:   rho *= 2.0; U /= 2.0         # 自适应 ρ
        elif r_dual > 10*r_prim: rho /= 2.0; U *= 2.0
        if max(r_prim, r_dual) < tol: break
    return {"W": Z, "r_prim": ..., "r_dual": ..., "converged": ...}
```

**验证**（M7）：原始残差 $1.58\times10^{1}\to1.57\times10^{-6}$、
对偶残差 $1.46\times10^{1}\to3.45\times10^{-7}$，**两者同时趋于 0**（跨 7 个数量级）；
并量化了乘性更新不动点的 KKT 残差 $\max|[\nabla f\circ W]|=2.8\times10^{5}$ 作为对照
（该值大，正说明"乘性更新的不动点不是所记录目标函数的驻点"）。

> **如实说明**：`converged=False` —— 严格容差 $10^{-7}$ 未在 400 轮内达到
> （对偶残差 $3.4\times10^{-7}$ 差一点）。这不影响"ADMM 提供可验证判据"这一结论，
> 但不应宣称"已收敛到容差"。

**这条改进的价值**：它是把这一线工作送进更高档位（NeurIPS/ICML/TPAMI）
**最省力的改造** —— 目标函数几乎不用动，只需换优化器就能获得：
①可验证的收敛判据；②自适应步长（对病态 $X^\top X$ 鲁棒）；③一套审稿人认账的理论叙述。

---

### 6.8 M8 — 自适应图 + 连通分量约束（对六篇"固定 knn 图"的补强，§5.4/§5.5）

**动机**：六篇的 $L^{(i)}$ 全部由 `construct_W(knn, heat_kernel, t=1.0)` 一次算死。
后果：①$k,t$ 是额外超参且对数据尺度敏感；②固定的 knn 图含大量**跨簇**边，
流形平滑会把不同类别的样本拉近；③**图不随特征选择过程演化**，而特征选择恰恰在改变数据几何。

**公式**（Nie et al. 2016 的自适应图学习 + Ky Fan 定理）：

$$
\begin{aligned}
\min_{S,F}\quad
&\sum_{i,j}\|x_i-x_j\|_2^2\,s_{ij}+\gamma\,\mathrm{Tr}(F^\top L_SF)\\
\text{s.t.}\quad
&s_i^\top\mathbf 1=1,\ s_{ij}\ge0,\ F^\top F=I,\ \mathrm{rank}(L_S)=n-c
\end{aligned}
\tag{6.10}
$$

**秩约束的等价转写**（这是该机制的数学核心）：

$$
\mathrm{rank}(L_S)=n-c
\iff \min_{F^\top F=I}\mathrm{Tr}(F^\top L_SF)=0
\iff \sum_{i=1}^{c}\lambda_i(L_S)=0
\tag{6.11}
$$

即"图恰好有 $c$ 个连通分量"—— 这正是**聚类结构**的定义。

**逐行闭式解**（给定 $F$ 后）：

$$
\min_{s_i}\ \tfrac12\Big\|s_i+\frac{d_i}{2\gamma}\Big\|^2
\ \text{s.t.}\ s_i^\top\mathbf 1=1,\ s_i\ge0
\quad\Longrightarrow\quad
s_i=\Pi_{\text{simplex}}\Big(-\frac{d_i}{2\gamma}\Big)
\tag{6.12}
$$

其中 $d_{ij}=\|x_i-x_j\|^2+\gamma\|f_i-f_j\|^2$，$\Pi_{\text{simplex}}$ 是**单纯形投影**
（Duchi et al. 2008，$O(n\log n)$ 闭式）。

**代码**：

```python
def adaptive_graph_learning(X, n_clusters, gamma=None, k=10, max_iter=100, ...):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    if gamma is None:                    # ← 关键：按数据尺度自动标定
        gamma = np.median(d2[np.isfinite(d2)]) / (2.0 * n_clusters / n)
    S = <knn 初始化>
    for it in range(max_iter):
        L_S = np.diag(S.sum(axis=1)) - S
        F = LA.eigh(L_S)[1][:, :n_clusters]           # F-update：Ky Fan
        G = ((F[:,None,:]-F[None,:,:])**2).sum(axis=2)
        D_sim = np.nan_to_num(d2 + gamma*G, nan=0., posinf=0.); np.fill_diagonal(D_sim, np.inf)
        for i in range(n):
            v = -D_sim[i] / (2.0*gamma); v[i] = 0.0
            S_new[i] = _simplex_projection(v)         # S-update：逐行单纯形投影
        S = (S_new + S_new.T) / 2.0
    evals = np.sort(LA.eigvalsh(np.diag(S.sum(1)) - S))
    return {"S": S, "F": F, "n_components": int((evals < 1e-8).sum()), ...}
```

**验证**（M8）：
* 自适应图恢复**恰好 4 个连通分量**（= 真值簇数）⇒ $\mathrm{rank}(L_S)=n-c$ 约束生效；
* $\gamma$ 的容差：在 $[10^{-3},10^{6}]\times\gamma^*$ 的 **9 个数量级**内均正确；
  过小（$10^{-6}\gamma^*$）退化为 1-NN 图（240 个分量）；
* 对照固定 knn 图：$t=0.1$ 时热核下溢（$\|L\|_F=0$）、$t=30$ 时近全连接，
  $\|L\|_F$ 跨 **7556 倍** ⇒ 直接解释了 §5.4 的权重塌缩。

**诚实说明**：自适应图**并非免超参**，而是把 2 个超参 $(k,t)$ 换成 1 个
**有量纲、可用 $\mathrm{median}(d^2)$ 自动标定**的 $\gamma$，并把结构约束变成
**可验证的** $\mathrm{rank}(L_S)=n-c$。这个表述比"免超参"更站得住脚。

---

### 6.9 八条思路的汇总

| 思路 | 针对的缺陷 | 类型 | 验证结果 |
|---|---|---|---|
| **M1** 修正加权 TNN | TOCL 管方向/偶数崩溃/切片缺失 | 修 bug + 语义澄清 | 秩结构与参考实现一致；$C\uparrow$ ⇒ rank $\downarrow$ |
| **M2** $c_m$ 闭式含义 | UGRFS $C$ 更新不可解释 | 化简 + 效应量量化 | 噪声识别 57.5%（基线 50%）；**弱效应 6.24%** |
| **M3** 标签感知视图权重 | §5.4 四篇共有缺陷 | **新机制** | $(0,1,0)\to(0.34,0.55,0.12)$ |
| **M4** 全票通过共识量化 | DHLI 互斥项语义未证 | 量化 | 软平均 2:1 vs $B$ 项 1:0 |
| **M5** $\Phi$ 软硬对照 + 带宽标定 | I²VSLC 二值化偏离 + 下溢 | 修 bug + 标定 | $\Phi$ 相差 342%；图差异 0.879 |
| **M6** 锚点双路径复现 | GRAFS 分解结构未讲透 | 复现 + 解释 | 误差单调降；一致性项精确归零 |
| **M7** ADMM + 收敛判据 | §5.7 六篇皆无收敛定理 | **新优化器** | $r_{\text{prim}},r_{\text{dual}}$ 同时趋于 0（跨 7 个数量级） |
| **M8** 自适应图 + 秩约束 | 六篇固定 knn 图 | **新机制** | 恰好恢复 $c$ 个分量；9 个数量级容差 |

---

## 7. 若要在此基础上再发一篇：定位建议

综合 §1.3 的 CCF 判据、§4 的演进脉络、§5 的缺陷清单、§6 的补强代码，
给出三个可行定位（按推荐度排序）。

### 定位 A（推荐）：把"视图权重"做成一个**可学习、尺度不变、标签感知**的量

**Motivation**（一句话）：现有 MVML 特征选择方法（UGRFS/GRAFS/EF²FS/I²VSLC 四篇同源）
用 $v_i\propto1/\mathrm{Tr}(G^\top L^{(i)}G)$ 估计视图重要性，该量**对嵌入尺度二次敏感、
与标签无关**，且在真实数据上因热核下溢而**塌缩到单一视图**（本文档首次量化）。

**Contribution**：
1. 形式化该缺陷（定理：$\mathrm{Tr}((cG)^\top L(cG))=c^2\mathrm{Tr}(G^\top LG)$ ⇒ 权重不具尺度不变性）；
2. 提出带**秩约束**的自适应视图图学习（M8）+ **标签感知能量**（M3），
   并证明 $\mathrm{rank}(L_S)=n-c$ 约束在单纯形投影下可达；
3. 用 ADMM 替换乘性更新（M7），给出**可验证的收敛判据**；
4. 实验：在 §6 的构造上展示基线塌缩 vs 本方法正确分配；在 6 个标准数据集上对比。

**目标venue**：AAAI / IJCAI（J1+J2 齐备）或 TPAMI / TKDE（若把定理做扎实）。

**风险**：需要补一个**真实数据集上的塌缩证据**（本节只给了合成数据）。

### 定位 B：统一"标签拓扑" —— 从 hybrid labels 到**可识别**的标签分解

**Motivation**：DHLI 与 TOCL 都把 $Y$ 拆成 $Y_c/Y_s/Y_n$ 并声称"与噪声模式无关"，
但该分解的**可识别性（identifiability）从未被讨论** —— 什么样的噪声模式下
$Y_c$ 能被唯一恢复？论文用 $\ell_1$ 稀疏 + "公共占多数"两个先验，但没给充要条件。

**Contribution**：给出 $Y_c/Y_s/Y_n$ 可识别的充分条件；构造反例说明现有目标函数
在何种噪声模式下会把 $Y_c$ 与 $Y_s$ 混淆；提出带可识别性保证的修正正则。

**目标venue**：AAAI / ICML（理论贡献型）。

**风险**：数学工作量最大，但从 §5 的偏差清单看，这条路的"空白"是真实存在的。

### 定位 C：把这一线的目标函数用 ADMM 重做一遍，附收敛定理

**Motivation**：§5.7 —— 六篇全部声称收敛、全部无定理，且实测震荡。

**Contribution**：对 UGRFS（结构最完整）写出 ADMM 分解，证明其收敛到 KKT 点；
实验上展示收敛性对超参鲁棒性、以及震荡消除带来的精度提升。

**目标venue**：比较稳妥的期刊（如 Information Sciences / PR），
因为"换优化器 + 补证明"属增量，但**证据链完整、风险最低**。

---

## 附录 A：复现命令

```bash
# 全部机制自检（8/8 通过）
.venv/Scripts/python.exe -m insight.run_demo

# 单机制（含详细数值）
.venv/Scripts/python.exe -m insight.run_demo M3
.venv/Scripts/python.exe -m insight.run_demo --list

# 仓库原有算法复现
bash run.sh --alg TOCL --data emotions
MVML_MAX_ITER=60 bash run.sh --alg TOCL --data emotions   # TOCL 每轮含 FFT+SVD，很慢
```

## 附录 B：本文档的方法论立场

本文档在以下三处刻意与"论文摘要式复述"不同：

1. **区分"论文声称"与"可验证"**。例如 DHLI 声称 "proven convergence"，
   但实测六篇均无定理；EF²FS 声称 embedded 优于 two-stage，
   但论文自认无法做删除式消融、也无二阶段基线 —— 这些都被如实记录。
2. **报告效应量而非仅方向**。例如 UGRFS 的样本置信度确实能识别噪声样本（方向对），
   但效应量只有 6.24%；M4 的 $B$ 项确实优于软平均（方向对），
   但优势是"1:0 vs 2:1"而非"能 vs 不能"。
3. **不把"优化器不收敛"误报为"机制无效"**。EF²FS 的 $A$ 训不出预期结构，
   根因是乘性更新无步长控制（§5.7），而非 embedded 融合的思想错误 ——
   这一点直接影响 §7 的定位选择。
