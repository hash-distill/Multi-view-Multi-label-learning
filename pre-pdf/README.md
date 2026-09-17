# pre-pdf — 论文收录说明

本目录存放与本仓库算法对应的论文全文，并记录每篇的来源与校验信息。
（项目英文原版 `README.md` 未做改动；本文件只描述本目录内容。）

## 收录总览

| 算法 | 论文 | 发表处 | 页数 | 全文 |
|---|---|---|---|---|
| **TOCL** | Tensor-based Opposing yet Complementary Learning for Multi-view Multi-label Feature Selection | ACM MM 2025, pp. 1822-1831 | 10 | ✅ |
| **UGRFS** | Uncertainty-Aware Global-View Reconstruction for Multi-View Multi-Label Feature Selection | AAAI 2025, 39(16) | 9 | ✅ |
| **EF2FS** | Embedded feature fusion for multi-view multi-label feature selection | Pattern Recognition 157 (2025) 110888 | 11 | ✅ |
| **DHLI** | Double-Layer Hybrid-Label Identification Feature Selection for Multi-View Multi-Label Learning | AAAI 2024, 38(11): 12295-12303 | 9 | ✅ |
| **I2VSLC** | Exploring view-specific label relationships for multi-view multi-label feature selection | Information Sciences 681 (2024) 121215 | 14 | ✅ |
| **GRAFS** | Anchor-guided global view reconstruction for multi-view multi-label feature selection | Information Sciences 679 (2024) 121124 | 13 | ✅ |
| **THBFS** | Scalable Multi-View Multi-Label Feature Selection via Tensor-Coupled Hypergraph-Bipartite Consensus | 未公开全文 | 1 | ⚠️ 仅宣传材料 |

**六篇已发表论文的全文现已齐备**；THBFS 仍只有一页宣传材料（且仓库中无其实现代码）。

## 文件清单

| 文件 | 来源 | 页数 | 大小 |
|---|---|---|---|
| `TOCL-2025-ACM-MM-Tensor-based-Opposing-yet-Complementary-Learning.pdf` | ACM MM 2025 | 10 | 2.62 MB |
| `AAAI-2025-UGRFS-Uncertainty-Aware-Global-View-Reconstruction.pdf` | AAAI 开放获取平台 | 9 | 1.20 MB |
| `EF2FS-2025-Pattern-Recognition-Embedded-feature-fusion.pdf` | Pattern Recognition 157 (2025) 110888 | 11 | 2.40 MB |
| `AAAI-2024-DHLI-Double-Layer-Hybrid-Label-Identification.pdf` | AAAI 开放获取平台 | 9 | 4.14 MB |
| `I2VSLC-2024-Information-Sciences-View-specific-label-relationships.pdf` | Information Sciences 681 (2024) 121215 | 14 | 2.15 MB |
| `GRAFS-2024-Information-Sciences-Anchor-guided-global-view-reconstruction.pdf` | Information Sciences 679 (2024) 121124 | 13 | 1.47 MB |
| `25-aaai-Uncertainty-Aware Global-View Reconstruction ....pdf` | 原有文件（作者版），与上表 UGRFS 同一篇 | 9 | 1.32 MB |
| `THBFS.pdf` | 原有文件，一页宣传材料 | 1 | 0.37 MB |
| `paper-notes/*.md` | 各篇题录、摘要、对应代码、变量对照 | – | – |

说明：`25-aaai-Uncertainty-Aware...pdf` 与
`AAAI-2025-UGRFS-...pdf` 是**同一篇论文的两个版本**（前者无 AAAI 抬头，推测为投稿/
作者版；后者为出版社版）。两个都保留，原有的那个未做改动。

## 内容核验方式

每份 PDF 都做过以下核验，确认是目标论文而非其他文件：

```bash
# 题名 / 作者 / 页数（与 Crossref 官方元数据比对）
pdfinfo "<file>.pdf" | grep -E "^Title|^Author|^Pages"

# 摘要与 DOI 页眉（Elsevier 版在 Subject 字段直接给出 DOI）
pdftotext -layout -f 1 -l 1 "<file>.pdf" - | head -20

# 文件完整性（sha256 与上传前一致）
sha256sum "<file>.pdf"
```

各篇的官方题录（作者、卷期、页码、DOI）来自 Crossref API，与 PDF 内页页眉一致：

| 算法 | DOI | Crossref 作者 |
|---|---|---|
| TOCL | 10.1145/3746027.3755447 | Pingting Hao, Huijie Zhang, Yongshan Zhang |
| UGRFS | 10.1609/aaai.v39i16.33876 | Pingting Hao, Kunpeng Liu, Wanfu Gao |
| EF2FS | 10.1016/j.patcog.2024.110888 | Pingting Hao, Wanfu Gao, Liang Hu |
| DHLI | 10.1609/aaai.v38i11.29120 | Pingting Hao, Kunpeng Liu, Wanfu Gao |
| I2VSLC | 10.1016/j.ins.2024.121215 | Pingting Hao, Weiping Ding, Wanfu Gao, Jialong He |
| GRAFS | 10.1016/j.ins.2024.121124 | Pingting Hao, Kunpeng Liu, Wanfu Gao |

## 版权说明

这些 PDF 由使用者提供并放入本仓库，**仅用于本地研究阅读**。
四篇期刊/会议论文（TOCL、EF2FS、I2VSLC、GRAFS）为出版社版权内容，请勿再分发；
两篇 AAAI 论文为 CC BY 开放获取，可自由分发。如需引用，请按各篇 DOI 正式引用。

## 与代码的对应关系

各论文的方法思路、代码位置、以及「论文符号 ↔ 代码变量」对照表，
见仓库根目录的 [`PAPERS_AND_CODE.md`](../PAPERS_AND_CODE.md)；
`paper-notes/` 下的单篇笔记给出更详细的题录与摘要。
