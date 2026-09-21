# 2023—2026 年多视图多标签学习新方法与研究方向综述

> **主题**：传统多视图多标签（Multi-View Multi-Label, MVML）问题如何借助大语言模型、
> Transformer、图神经网络、扩散模型、对比学习、变分编码器和自动编码器等新方法解决。
>
> **检索时间**：2026-09-18  
> **时间范围**：主体为 2024—2026 年，补充两篇具有方法源头意义的 2023 年论文。  
> **范围说明**：本文优先收录明确研究 MVML/MvMLC/MVMLFS 的论文，不把“仅多视图”或
> “仅多标签”的工作混入核心清单。论文方法与发表信息以会议、期刊或 PMLR/OpenReview
> 等论文页面为准。

---

## 1. 核心结论

近三年的 MVML 研究已经从低秩矩阵分解、固定共享子空间和浅层图正则，逐渐转向：

1. **视图与标签双缺失建模**：同时处理缺失视图、缺失标签和弱监督，而非只解决其中之一；
2. **深层语义表示学习**：通过共享—私有编码器、信息瓶颈、对比学习和标签原型获得任务相关表示；
3. **生成式视图补全**：利用自动编码器、变分编码器或扩散模型恢复缺失视图；
4. **标签感知的动态融合**：不再假设所有标签都同等依赖所有视图，而是显式学习 view-label 匹配；
5. **高阶结构建模**：由普通样本图、标签图扩展至异构图和超图；
6. **半监督与可信伪标签**：利用未标注样本，并显式估计伪标签的不确定性；
7. **LLM 语义先验**：用大语言模型补充传统统计关系难以表达的特征、视图和标签语义。

在本轮检索范围内，**直接使用大语言模型解决标准 MVML 问题的正式论文仍然很少**。
目前最明确的代表是 AAAI 2026 的 LLM+GNN 多视图多标签特征选择方法。多数新工作仍属于
Transformer、GNN、扩散模型、对比学习和编码器—解码器路线。因此，LLM 与 MVML 的结合
仍有较大的研究空间。

---

## 2. 推荐论文速查表

| 优先级 | 年份 | 论文/方法 | 发表处 | 传统问题 | 新技术 | 推荐理由 |
|---|---:|---|---|---|---|---|
| S | 2026 | LLM Semantic Reasoning + GNN | AAAI | MVML 高维特征选择过度依赖统计关系 | LLM 语义评分、异构图、GAT | 本轮检索中最直接的 LLM+MVML 工作 |
| S | 2026 | PCVE | ICLR | 视图和标签同时缺失；跨视图过度对齐 | 变分编码、信息瓶颈、置换一致性 | 理论性强，适合发展概率表示与任意缺失模式 |
| S | 2026 | CTRL | TPAMI | 双缺失条件下伪标签不可靠 | 紧凑语义、证据神经网络、D-S 理论 | 将表示学习与标签不确定性统一起来 |
| S | 2025 | Compact Semantic Information | ICML | 有限监督与数据双缺失 | 跨视图互信息、冗余压缩、软伪标签交叉补全 | 顶会强基线，信息论路线清晰 |
| A | 2026 | HyperAHSF | AAAI | 普通图只能表达成对关系 | 超图神经网络、标签原型对比 | 适合研究视图—标签高阶交互 |
| A | 2026 | RMAN-MMFS | AAAI | 视图内、视图间冗余特征 | 多头注意力、跨头注意力、动态冗余约束 | 面向深度 MVML 特征选择的最新代表 |
| A | 2025 | Diffusion-Guided Redundancy Removal | AAAI | 缺失视图及冗余视图 | 条件扩散、伪标签、视图筛选 | 把生成式补全和冗余移除结合起来 |
| A | 2025 | SMVTEP | AAAI | 大量无标签样本、视图能力不均衡 | GAN、对比学习、Transformer、Mixup | 半监督 MVML 模块较完整 |
| A | 2025 | VAMS | AAAI | 统一融合忽略标签对视图的不同依赖 | view-label matching graph、GNN | 重新定义 MVML 的融合与决策过程 |
| A | 2025 | TACVI-Net | Neural Networks | 缺失视图补全含有任务无关噪声 | 信息瓶颈、视图编码器、自动编码器 | 编码器—解码器结构清晰，便于扩展 |
| A | 2024 | Semantic Invariance + Prototype | ICML | 部分视图和部分标签不可用 | 信息瓶颈、语义不变性、标签原型 | 现代不完整 MVML 的重要强基线 |
| A | 2024 | AIMNet | AAAI | 原始空间补全噪声较大 | 嵌入空间联合注意力补全、标签 GAT | 适合潜空间补全和可信融合研究 |
| B+ | 2024 | LSGC | Neural Networks | 视图与标签双缺失 | 标签 GCN、样本—标签对比、伪标签 | 代表 label-aware contrastive 路线 |
| B+ | 2024 | DIMvSML | ACM MM | 不完整视图和大量未标注样本 | 深图网络、对比学习、无偏风险估计 | 适合研究半监督风险校正 |
| 基础 | 2023 | MTD | NeurIPS | 不完整视图与弱多标签 | 双通道解耦、Mask、对比与图正则 | 共享—私有表示解耦的重要起点 |
| 基础 | 2023 | Label-Guided Masked Transformers | AAAI | 任意视图/标签缺失和标签相关性 | 双 Transformer、Mask、自适应融合 | Transformer 型深度 MVML 的代表起点 |

> **优先级含义**：S 表示兼具新颖性、发表质量和继续研究空间；A 表示值得作为主要基线或
> 方法模块；B+ 表示适合补充某一技术路线；“基础”表示理解近年演进所需的源头工作。

---

## 3. 重点论文解读

### 3.1 LLM 语义推理 + GNN：首次直接切入 MVML 特征选择

**论文**：*Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-View
Multi-Label Feature Selection*  
**发表处**：AAAI 2026  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/39180)  
**DOI**：`10.1609/aaai.v40i25.39180`

#### 解决的问题

传统 MVML 特征选择通常依据相关系数、流形结构、低秩性或重构误差评价特征，难以使用
“猫—耳朵—视觉视图”这类自然语言层面的语义知识，尤其在小样本条件下统计关系不稳定。

#### 方法

- 让 LLM 作为评估代理，对 feature、view 和 label 描述之间的潜在语义相关性打分；
- 建立同时包含语义关系和统计关系的两层异构图；
- 使用轻量 GAT 学习节点表示，并将特征节点的显著性作为排序依据。

#### 研究价值与局限

这是最值得优先跟进的“大模型+MVML”论文。但它主要处理**特征选择**，LLM 也主要作为
离线语义评分器，并未形成端到端的 LLM-MVML 分类系统。语义评分的稳定性、成本、领域偏差、
置信度校准以及无自然语言特征描述时的处理方式，都是可继续研究的问题。

---

### 3.2 PCVE：置换一致性变分编码

**论文**：*Permutation-Consistent Variational Encoding for Incomplete Multi-View
Multi-Label Classification*  
**发表处**：ICLR 2026  
**链接**：[OpenReview](https://openreview.net/forum?id=y4LyiOIOUn)

#### 解决的问题

在视图与标签同时缺失时，不同视图的编码结果可能不一致；若强制简单对齐，又可能把视图私有
噪声带入共享表示，形成过度对齐。

#### 方法

- 用变分编码器学习各视图的潜在分布；
- 通过 evidence lower bound 保留任务相关信息；
- 用 permutation-consistent regularization 约束不同视图对同一目标语义的分布一致性；
- 使用 masked multi-label objective 只利用可观测标签监督；
- 在不显式恢复原始输入的情况下，对缺失视图进行稳健推断。

#### 研究价值

PCVE 将不完整 MVML 解释为概率信息整合问题，适合继续发展：

- Product-of-Experts / Mixture-of-Experts 多视图后验；
- 任意视图集合输入；
- 不确定性感知的融合；
- 生成式缺失视图推断；
- 面向真实图像、文本、音频视图的深层编码器替换。

---

### 3.3 CTRL：可信伪标签与不确定性

**论文**：*Learning Compact Semantic Information and Reliable Pseudo-Labels for
Incomplete Multi-View Multi-Label Classification*  
**发表处**：IEEE TPAMI 2026  
**链接**：[IEEE Xplore](https://ieeexplore.ieee.org/document/11397552)  
**DOI**：`10.1109/TPAMI.2026.3665813`

#### 方法与价值

- 压缩任务无关冗余，学习高纯度的跨视图共享表示；
- 通过 Beta Evidential Neural Network 对标签分布建模；
- 使用 Dempster–Shafer 理论融合证据；
- 根据不确定性和 belief mass 生成更可靠的伪标签。

该工作的重要意义不是“增加一个伪标签损失”，而是把**表示是否可靠**和**标签是否可信**纳入
统一框架。它适合作为可信 MVML、风险敏感 MVML 和生成式补全质量控制的参考。

---

### 3.4 跨视图互信息与紧凑语义表示

**论文**：*Learning Compact Semantic Information for Incomplete Multi-View Missing
Multi-Label Classification*  
**发表处**：ICML 2025  
**链接**：[PMLR](https://proceedings.mlr.press/v267/wen25c.html)

#### 方法

- 从共享信息对下游任务的充分性出发，压缩任务无关冗余；
- 追求跨视图互信息最大化，而不只做普通成对对比；
- 用双分支软伪标签进行交叉补全，缓解标签缺失。

#### 研究价值

这篇论文适合与 PCVE、ICML 2024 的语义不变表示论文连续阅读，用于理解不完整 MVML 中的
“一致性、充分性和紧凑性”并不完全等价。

---

### 3.5 HyperAHSF：高阶语义超图

**论文**：*Hypergraph-Based Multi-View Multi-Label Classification via Adaptive
High-Order Semantic Fusion*  
**发表处**：AAAI 2026  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/39719)  
**DOI**：`10.1609/aaai.v40i30.39719`

#### 方法

- 在每个视图内部建立表达群组级相似性的 view-specific hyperedges；
- 用跨视图超边连接同一样本的多视图表示；
- 通过超图神经网络分别传播私有信息和共识信息；
- 对共识表示施加标签驱动的原型对比损失。

#### 研究价值

普通图通常依赖成对边，需要多层传播才能间接表达多个样本、视图和标签之间的高阶关系。
超图路线适合与标签原型、LLM 语义图和动态结构学习结合。

---

### 3.6 RMAN-MMFS：多头注意力特征选择

**论文**：*Redundancy-Optimized Multi-Head Attention Networks for Multi-View
Multi-Label Feature Selection*  
**发表处**：AAAI 2026  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/39571)  
**DOI**：`10.1609/aaai.v40i28.39571`

#### 方法

- 每个注意力头建模一个视图内部的特征关系；
- 通过不同头之间的 cross-attention 捕获跨视图互补性；
- 静态冗余项抑制单视图内部冗余；
- 动态冗余项建模已选特征与未选特征之间的关系。

#### 研究价值

它与 LLM+GNN 论文都属于 MVMLFS，但分别强调**数据驱动的注意力关系**和**LLM 提供的语义
关系**。二者结合可形成“语义先验+统计注意力+冗余控制”的自然研究路线。

---

### 3.7 扩散模型补全与冗余视图移除

**论文**：*Incomplete Multi-View Multi-Label Classification via Diffusion-Guided
Redundancy Removal*  
**发表处**：AAAI 2025  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/34176)  
**DOI**：`10.1609/aaai.v39i18.34176`

#### 方法

- 用伪标签作为条件训练扩散模型，恢复缺失视图；
- 根据视图新增信息量和样本分类难度判断某视图是否冗余；
- 在恢复缺失信息的同时，避免无效视图持续参与融合。

#### 研究价值与风险

该方法开辟了“生成式补全+视图筛选”的路线，但生成视图可能出现语义幻觉。后续研究不能只
比较最终分类准确率，还应单独评估生成视图的一致性、置信度及其对错误标签的放大作用。

---

### 3.8 SMVTEP：半监督 Transformer 与增强伪标签

**论文**：*Semi-Supervised Multi-View Multi-Label Learning with View-Specific
Transformer and Enhanced Pseudo-Label*  
**发表处**：AAAI 2025  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/34028)  
**DOI**：`10.1609/aaai.v39i17.34028`

#### 方法

- GAN 提取共享和私有表示；
- 对抗机制与信息论对比学习约束一致性和差异性；
- Transformer 融合视图置信度、标签语义和标签依赖；
- 使用 Mixup 与类别增强伪标签利用未标注样本。

#### 研究价值

适合用于研究“少量标注+大量未标注”的 MVML。其模块较多，后续复现时应通过消融实验验证
各组件的独立贡献，避免把性能提升简单归因于 Transformer。

---

### 3.9 VAMS：View-Label 匹配选择

**论文**：*Multi-View Multi-Label Classification via View-Label Matching Selection*  
**发表处**：AAAI 2025  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/35447)  
**DOI**：`10.1609/aaai.v39i20.35447`

#### 方法

- 将一个对象视为多个 view 构成的 bag；
- 分别建立对象图和标签图；
- 将各视图节点连接到标签节点，形成统一 view-label matching graph；
- 用图网络更新节点与边，输出每一个 view-label 匹配分数。

#### 研究价值

VAMS 不再遵循“先把视图压成一个向量，再统一预测标签”的固定流程，而是显式回答“哪个标签
应当使用哪个视图”。它适合进一步改造成跨注意力、动态图或稀疏 Mixture-of-Experts。

---

### 3.10 TACVI-Net：任务增强的跨视图补全

**论文**：*Task-Augmented Cross-View Imputation Network for Partial Multi-View
Incomplete Multi-Label Classification*  
**发表处**：Neural Networks 187 (2025), 107349  
**链接**：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S089360802500228X)  
**DOI**：`10.1016/j.neunet.2025.107349`

#### 方法

1. 用信息瓶颈指导的 view-specific encoder-classifier 提取与任务高度相关的表示；
2. 用 autoencoder 型多视图重建网络恢复缺失信息；
3. 使用重建后的多视图语义支持最终分类。

#### 研究价值

它将“先提纯任务信息，再补全缺失视图”与直接重建原始特征区分开来，结构清晰，适合替换成
VAE、扩散解码器或 foundation-model encoder。

---

### 3.11 语义不变表示与标签原型

**论文**：*Partial Multi-View Multi-Label Classification via Semantic Invariance
Learning and Prototype Modeling*  
**发表处**：ICML 2024  
**链接**：[PMLR](https://proceedings.mlr.press/v235/liu24bv.html)

#### 方法与价值

- 通过信息瓶颈压缩非共享信息并保留任务相关共享信息；
- 在潜空间中建立多标签原型；
- 数据驱动地学习标签相关性；
- 同时兼容部分数据与完整数据。

该工作是理解 ICML 2025、PCVE 以及后续紧凑语义表示路线的重要中间节点。

---

### 3.12 AIMNet：嵌入空间缺失实例补全

**论文**：*Attention-Induced Embedding Imputation for Incomplete Multi-View Partial
Multi-Label Classification*  
**发表处**：AAAI 2024  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/29293)  
**DOI**：`10.1609/aaai.v38i12.29293`

#### 方法

- 不在原始空间或核空间恢复视图，而是在高层 embedding 空间补全；
- 用跨视图联合注意力估计缺失实例表示；
- 根据注意力置信度对补全后的视图动态加权；
- 用标签相关矩阵和 GAT 指导标签特定表示学习。

#### 研究价值

适合与扩散补全进行对比：AIMNet 是确定性的潜空间补全，扩散模型是条件生成式补全，两者可以
在补全准确性、分类性能、计算开销和不确定性方面系统比较。

---

### 3.13 LSGC：标签语义引导的对比学习

**论文**：*Deep Dual Incomplete Multi-View Multi-Label Classification via Label
Semantic-Guided Contrastive Learning*  
**发表处**：Neural Networks 180 (2024), 106674  
**链接**：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608024005987)  
**DOI**：`10.1016/j.neunet.2024.106674`

#### 方法

- DNN 提取高层多视图特征；
- GCN 捕获标签语义和标签相关性；
- 设计 sample-label contrastive loss；
- 对缺失标签采用伪标签填充和置信加权。

#### 研究价值

相比只做 sample-sample 对比，它显式把标签语义引入对比目标，可作为 label-aware contrastive
learning 的代表基线。

---

### 3.14 DIMvSML：不完整半监督学习与无偏损失

**论文**：*Deep Incomplete Multi-View Semi-Supervised Multi-Label Learning Network
with Unbiased Loss*  
**发表处**：ACM Multimedia 2024  
**链接**：[OpenReview 论文页面/PDF](https://openreview.net/pdf?id=5XwylUAmnY)  
**DOI**：`10.1145/3664647.3681414`

#### 方法

- 深图网络利用结构相似性恢复特征；
- 结构特定的深层特征提取器保留各视图信息；
- 用实例级对比损失保持跨视图一致性；
- 用无偏损失修正半监督多标签风险估计，并使用伪标签增强训练。

#### 研究价值

其突出价值是把表示学习之外的**风险估计偏差**纳入研究，可用于对比常规伪标签自训练是否
因为选择偏差而产生虚假的性能提升。

---

## 4. 两篇建议补读的 2023 年源头工作

### 4.1 MTD：Masked Two-Channel Decoupling

**论文**：*Masked Two-Channel Decoupling Framework for Incomplete Multi-View Weak
Multi-Label Learning*  
**发表处**：NeurIPS 2023  
**链接**：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/66772e6aa61e54ae16443ae1d78a7319-Abstract-Conference.html)

核心贡献：

- 把单通道表示解耦为共享表示和视图私有表示；
- 用跨通道对比损失强化两类表示的语义性质；
- 用标签引导的图正则保留样本几何结构；
- 对向量特征执行随机片段 Mask；
- 兼容任意视图缺失和标签缺失。

### 4.2 Label-Guided Masked Transformers

**论文**：*Incomplete Multi-View Multi-Label Learning via Label-Guided Masked View-
and Category-Aware Transformers*  
**发表处**：AAAI 2023  
**链接**：[AAAI 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/26060)  
**DOI**：`10.1609/aaai.v37i7.26060`

核心贡献：

- 用 view-aware Transformer 聚合跨视图信息；
- 用 category-aware Transformer 学习类别与子类别表示；
- 自适应学习不同视图的融合权重；
- 用标签流形约束提升监督信息利用率；
- 在模型设计上直接考虑视图与标签不完整。

---

## 5. 最值得继续研究的五个方向

### 5.1 LLM 语义先验 + 不确定性校准 + MVML

#### 核心想法

让 LLM 提供 feature-view-label、view-view 或 label-label 的语义先验，再由数据驱动网络学习
这些先验在当前数据集中的可信程度，而不是直接把 LLM 分数当作真值。

#### 可行框架

```text
特征/视图/标签描述
        │
        ▼
LLM 或领域大模型 ──► 语义关系图 + 语义置信度
        │
        ├──────────────┐
        ▼              ▼
统计关系图        不确定性校准模块
        └──────┬───────┘
               ▼
       GNN / Hypergraph / Transformer
               ▼
       特征选择或多标签分类
```

#### 关键研究问题

- LLM 语义评分在不同提示词、模型和温度下是否稳定？
- 语义先验与数据统计关系冲突时如何融合？
- 没有自然语言特征名时，如何生成可用的特征描述？
- 是否可用本地开源模型、知识图谱或 RAG 降低成本并增强领域知识？
- LLM 先验是否真正提升跨数据集泛化，而不只是拟合标签名称？

#### 推荐起点

AAAI 2026 LLM+GNN、AAAI 2026 HyperAHSF、AAAI 2026 RMAN-MMFS。

---

### 5.2 生成式缺失视图恢复与“幻觉视图”抑制

#### 核心想法

使用 VAE、latent diffusion 或条件扩散模型恢复缺失视图，同时预测补全结果的不确定性；当补全
置信度不足时，应降低该视图的融合权重或完全跳过补全。

#### 可研究模块

- 标签条件与跨视图条件生成；
- classifier-guided / classifier-free guidance；
- 补全视图与真实视图的一致性判别器；
- evidential uncertainty 或 ensemble uncertainty；
- 按样本、视图和标签动态调整补全贡献；
- 直接比较“补全后分类”和“不补全但鲁棒编码”两条路线。

#### 推荐起点

AAAI 2025 Diffusion-Guided、ICLR 2026 PCVE、TPAMI 2026 CTRL、Neural Networks
2025 TACVI-Net、AAAI 2024 AIMNet。

---

### 5.3 标签条件的 View-MoE 与动态路由

#### 核心想法

每个视图由独立 encoder/expert 建模，每个标签使用自己的 query，通过 cross-attention 或稀疏
路由选择最有用的视图，避免全局固定权重。

#### 可行结构

```text
View 1 ─► Encoder/Expert 1 ─┐
View 2 ─► Encoder/Expert 2 ─┼─► Label-conditioned Router ─► 多标签预测
  ...                       │          ▲
View V ─► Encoder/Expert V ─┘          │
                               标签嵌入/标签原型
```

#### 可研究问题

- 路由权重应按样本、标签还是样本—标签二元组计算？
- 缺失视图时如何动态重路由？
- 如何避免所有标签塌缩到同一强视图？
- 能否结合稀疏 MoE 降低多视图编码计算量？

#### 推荐起点

AAAI 2025 VAMS、AAAI 2025 SMVTEP、AAAI 2023 Label-Guided Transformers。

---

### 5.4 超图/图 Transformer 的高阶标签关系建模

#### 核心想法

从 label-label 成对相关性提升到标签群组、样本群组和跨视图群组关系，使用动态超图或图
Transformer 学习结构。

#### 可研究组合

- 标签原型作为图节点；
- LLM 生成语义超边，数据统计生成结构超边；
- 用门控机制融合语义图和统计图；
- 动态更新超边而不是使用固定 KNN；
- 对超图结构增加稀疏性和可解释性约束。

#### 推荐起点

AAAI 2026 HyperAHSF、AAAI 2025 VAMS、AAAI 2026 LLM+GNN、Neural Networks
2024 LSGC。

---

### 5.5 从人工特征 benchmark 迁移到真实基础模型视图

#### 问题

不少 MVML 论文仍使用 Corel5k、Pascal07、ESPGame、IAPRTC12、MIRFlickr 等经典数据，
并把 HOG、SIFT、颜色直方图或传统文本特征作为不同视图。这有利于横向比较，却不能充分证明
方法适用于现代多模态数据。

#### 推荐实验设计

- 图像视图：CLIP、DINOv2、SigLIP 等不同视觉编码器；
- 文本视图：现代文本编码器或领域 LLM；
- 音频视图：CLAP、AudioMAE 等；
- 比较冻结编码器、参数高效微调和完全微调；
- 分别模拟视图随机缺失、结构化缺失、标签缺失和模态噪声；
- 验证模型是否对不同编码器和不同缺失机制稳健。

该方向的研究价值在于判断：论文性能来自真正的 MVML 机制，还是来自老数据集上的特定人工
特征结构。

---

## 6. 推荐阅读顺序

### 6.1 建立基础

1. AAAI 2023 Label-Guided Masked Transformers；
2. NeurIPS 2023 MTD；
3. ICML 2024 Semantic Invariance + Prototype Modeling。

目标：理解 Mask、Transformer、共享—私有解耦、信息瓶颈和标签原型如何进入 MVML。

### 6.2 理解缺失数据的不同处理路线

4. AAAI 2024 AIMNet：潜空间补全；
5. Neural Networks 2024 LSGC：标签语义对比；
6. ACM MM 2024 DIMvSML：半监督无偏风险；
7. Neural Networks 2025 TACVI-Net：编码器—解码器补全；
8. AAAI 2025 Diffusion-Guided：扩散生成补全。

目标：比较跳过缺失视图、确定性补全、概率编码和生成式补全的差异。

### 6.3 进入当前前沿

9. ICML 2025 Compact Semantic Information；
10. AAAI 2025 VAMS；
11. AAAI 2025 SMVTEP；
12. ICLR 2026 PCVE；
13. TPAMI 2026 CTRL；
14. AAAI 2026 HyperAHSF；
15. AAAI 2026 RMAN-MMFS；
16. AAAI 2026 LLM+GNN。

目标：形成“信息论表示—可信伪标签—高阶结构—LLM 语义先验”的完整认识。

---

## 7. 后续实验建议

### 7.1 数据设置

至少同时报告：

- 完整视图、完整标签；
- 仅视图缺失；
- 仅标签缺失；
- 视图与标签双缺失；
- 视图质量下降或带噪；
- 随机缺失与非随机/结构化缺失。

不同方法应使用相同数据划分、相同缺失掩码和相同预训练特征，避免把数据差异误判为方法提升。

### 7.2 指标

常见 MVML 指标包括：

- Average Precision；
- Hamming Loss；
- Ranking Loss；
- One-Error；
- Coverage；
- AUC 或 Macro/Micro F1。

除平均值外应报告标准差，并采用多数据集统计检验。生成式补全方法还应单独报告补全质量、
不确定性校准和分类收益之间的关系。

### 7.3 必做消融

- 共享编码器与视图专用编码器；
- 不同融合位置：early、intermediate、late fusion；
- 无 LLM 先验、随机先验、真实 LLM 先验；
- 无标签图、固定标签图、动态标签图；
- 无补全、确定性补全、生成式补全；
- 无伪标签、硬伪标签、软伪标签、不确定性过滤伪标签；
- 固定视图权重与样本—标签级动态权重。

### 7.4 大模型方向的额外验证

- 更换 LLM 后语义关系是否稳定；
- 不同 prompt 的均值和方差；
- 是否存在标签名称泄漏；
- 去除自然语言描述后性能变化；
- LLM 推理成本和可重复性；
- 使用开源本地模型能否复现闭源模型结果；
- 语义先验错误时模型是否能自动降低其权重。

---

## 8. 选题建议

综合创新性、可实现性与当前文献空白，推荐以下三个题目方向：

### 方向 A：不确定性校准的 LLM 引导 MVML

> 使用 LLM 构建 feature-view-label 语义图，通过证据神经网络估计 LLM 先验可信度，
> 再与数据驱动统计图联合进行多视图多标签特征选择或分类。

优点：直接命中 LLM+MVML 空白；能够复用现有 MVMLFS 代码和评价体系。  
风险：必须严格控制 LLM 输出稳定性、成本和标签语义泄漏。

### 方向 B：面向双缺失 MVML 的可信扩散补全

> 用标签条件 latent diffusion 恢复缺失视图，通过 evidential uncertainty 判断补全是否可靠，
> 并在样本—标签层面动态决定使用、降权或拒绝生成视图。

优点：结合扩散模型、缺失视图和可信学习，问题明确。  
风险：训练开销较大，必须证明补全本身有效，而不是仅依赖分类器容量。

### 方向 C：标签条件的多视图稀疏专家路由

> 为每个视图建立专家编码器，用标签原型作为 query，在样本—标签层面执行稀疏路由；
> 缺失视图时自动重分配专家权重。

优点：结构直观、可解释，能自然处理不同标签依赖不同视图。  
风险：需要防止路由塌缩，并控制专家数量带来的计算开销。

---

## 9. 最小必读清单

如果时间有限，优先阅读以下 8 篇：

1. [LLM Semantic Reasoning + GNN，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39180)
2. [PCVE，ICLR 2026](https://openreview.net/forum?id=y4LyiOIOUn)
3. [CTRL，TPAMI 2026](https://ieeexplore.ieee.org/document/11397552)
4. [HyperAHSF，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39719)
5. [Compact Semantic Information，ICML 2025](https://proceedings.mlr.press/v267/wen25c.html)
6. [Diffusion-Guided Redundancy Removal，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34176)
7. [VAMS，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35447)
8. [Semantic Invariance + Prototype Modeling，ICML 2024](https://proceedings.mlr.press/v235/liu24bv.html)

---

## 10. 证据边界

- 本文是面向选题的定向检索与方法归纳，不等同于系统综述或穷尽性检索；
- “LLM+MVML 工作稀少”指本轮以精确 MVML 术语检索到的正式论文数量较少，不表示不存在
  所有邻近的多模态、多标签或基础模型研究；
- 论文作者报告的领先结果需要在统一数据划分、缺失掩码、特征和评价实现下重新验证；
- 经典数据集上的结果不能直接证明方法在真实多模态基础模型特征上同样有效；
- 对扩散补全、伪标签和 LLM 先验，应同时考察错误传播与不确定性，而不只报告最终分类指标。

