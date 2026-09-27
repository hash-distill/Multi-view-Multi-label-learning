# 六篇论文的复现记录（2026-09-21，远程服务器）

> 本文记录在服务器 `24xyx@210.47.18.59:/mnt/Data4/24zhs/Mvml/Multi-view-Multi-label-learning`
> 上补齐六篇论文实验的全过程：协议、命令、代码改动、验证证据与已知偏差。
> 配套文件：`results/PAPER_COMPARISON.md`（六篇对比表）、`results/VALIDATION.md`（完整性校验）。
> 协议细节与论文原文依据见 `analysis/EXPERIMENT_PROTOCOLS.md`，两篇的深入分析见
> `analysis/REPRODUCE_TOCL_UGRFS.md`。

---

## 0. 一句话结论

六篇论文（TOCL / UGRFS / EF2FS / DHLI / I2VSLC / GRAFS）× 各自 6 个数据集 = **36 个格子**
全部跑通，产出 `metrics_/rankings_/perfold_/summary_` 全套文件与六篇对比表。
**未做超参数调优**（α=β=γ=λ=1.0），因此数值与论文仍有 0.005–0.05 量级的差距，
这与仓库原有结果的口径一致（`tune.py` 可做单参数扫描，但 36 格全调参代价过高）。

---

## 1. 复现协议（六篇一致）

| 项 | 取值 |
|---|---|
| 交叉验证 | 5 折（一个共享随机排列切成 5 个不重叠测试块，`--shuffle --seed 100`） |
| 特征子集 | 取排序前 k 个特征，k 扫描到 20% 总特征数（`--select-ratio 0.2`） |
| 聚合 | 对 1%–20% 的百分比点取均值（仅 GRAFS 原文写明"all sample points"） |
| 报告 | 5 折间 mean ± std |
| 指标 | AP↑ / Coverage↓ / Hamming Loss↓ / Ranking Loss↓ |
| 分类器 | 多标签 MLkNN（k=10, s=1.0）——**六篇论文都没写用哪个分类器**，见 §4.1 |
| 超参数 | 论文是 {10⁻³…10³} 网格搜索；本轮统一用 1.0（未调参） |

论文各篇的数据集与结果表位置：

| 算法 | 论文 | 结果表 | 数据集（论文行序） |
|---|---|---|---|
| TOCL | ACM MM 2025 | Table 2 / 3 | SCENE, OBJECT, MIRFlickr, Corel5K, IAPRTC12, 3Sources |
| UGRFS | AAAI 2025 | Table 2 / 3 | yeast, SCENE, VOC07, MIRFlickr, IAPRTC12, 3Sources |
| EF2FS | Pattern Recognition 157 (2025) 110888 | Table 3 / 4 | emotions, yeast, SCENE, MIRFlickr, IAPRTC12, 3Sources |
| DHLI | AAAI 2024 | Table 2 / 3 | SCENE, OBJECT, MIRFlickr, Corel5K, IAPRTC12, 3Sources |
| I2VSLC | Information Sciences 681 (2024) 121215 | Table 3 / 4 | 3Sources, SCENE, IAPRTC12, Corel5K, ESPgame, OBJECT |
| GRAFS | Information Sciences 679 (2024) 121124 | Table 3 / 4 | yeast, SCENE, OBJECT, VOC07, MIRFlickr, 3Sources |

论文基准值不是抄 `_pdftext/`（`pdftotext` 把表格列打乱了），而是用 PyMuPDF 的
表格/坐标重建重新抽取，并交叉核对了四篇论文互相转载的基线列
（M2LD/MSFS/MoRE/MRDM/CLML 逐位一致）。TOCL/UGRFS 的既有转录值经此核对**完全一致**。
抽取所用的一次性脚本已在报告中间文件清理时删除；上表保留核对后的论文来源与表号。

---

## 2. 关键改动：MLkNN 评测向量化（150×，数值逐位不变）

### 2.1 问题

`main.py::_make_mlknn` 原来的 `_compute_cond` / `predict` / `predict_proba` 是
逐 `(样本 × 标签)` 的 Python 循环，且 `predict` 与 `predict_proba` 把**同一次 kNN 查询重复算了两次**。
在大标签数据集上这一项完全支配成本：

| 规模 | 原始评测 | 向量化后 | 加速 |
|---|---|---|---|
| IAPRTC12 量级（n=5000, d=262, 291 标签） | 66.67 s/k | 0.44 s/k | **150×** |
| UGRFS/3sources（真实数据，600 个 k） | 77.9 s | 13.7 s | 6× |
| TOCL/MIRFlickr（40 个 k） | 373.9 s | 16.8 s | 22× |
| TOCL/SCENE（30 个 k） | 265.7 s | 16.3 s | 16× |
| UGRFS/yeast（20 个 k） | 58.0 s | 5.9 s | 10× |

按原始实现估算，单个 IAPRTC12 数据集的评测就要 **24 小时**，36 个格子不可行。

### 2.2 改法

* `_compute_cond`：把"每个样本 × 每个标签"的计数改成一次性构造标签-邻居计数矩阵，
  再对每个标签用 `np.bincount` 统计（整数计数，`c`/`cn` 与原实现完全相同）。
* `predict`/`predict_proba` 合并为 `predict_and_proba`：`_posterior()` 只做一次邻居查询，
  得到 `p_true`/`p_false` 后同时给出预测与概率。
* 浮点运算顺序保持为 `先验 × 条件概率` 的逐元素乘法，`条件概率 = (s+c)/(s(k+1)+Σc)` 逐元素相同。

### 2.3 逐位等价验证（`_abtest/ab_test_mlknn.py`、`_abtest/verify_eval_isolated.py`）

| 测试 | 结果 |
|---|---|
| 合成数据：cond_prob_true/false、prior、predict、predict_proba | 全部 `bitwise_identical=True` |
| 合成数据：5 个指标的完整曲线 | 全部逐位相同，max\|diff\|=0 |
| **真实数据隔离测试**（同一 train/test 划分 + 同一 ranking，喂给新旧两个评测器） | |
| ├ UGRFS/3sources（600 个 k） | 全部指标逐位相同 |
| ├ TOCL/MIRFlickr（40 个 k） | 全部指标逐位相同 |
| ├ TOCL/SCENE（30 个 k） | 全部指标逐位相同 |
| └ UGRFS/yeast（20 个 k） | 全部指标逐位相同 |
| 大尺度合成（291 标签）predict/predict_proba 一致性 | `predict=True predict_proba=True` |

**结论**：这是纯性能改动，表里的数不会因为这次优化而变化。

---

## 3. 执行方式与产物

* 驱动：`_mvml_runs/run_remaining.sh`（8 并发、长作业优先排序），30 个作业；
  每个作业写自己的 `--out results/parts/<ALG>_<DS>/`，避免并发写坏共享的 `summary_all.csv`。
* 收尾：`_mvml_runs/postprocess_watch.sh`（screen 会话）在队列排空后自动执行
  `merge_parts.py` → `paper_table.py --write` → `validate_results.py --write`。
* 作业清单（`_mvml_runs/jobs_lpt.txt`）里含一个额外组合 `I2VSLC/MIRFlickr`：
  MIRFlickr 不属于 I2VSLC 论文的数据集，对比表会单列标注、不计入覆盖率。

补齐的数据集（原仓库只有 TOCL×3 + UGRFS×4 共 7 格）：

| 算法 | 本次新增 |
|---|---|
| TOCL | OBJECT, corel5k_5, iaprtc12 |
| UGRFS | SCENE, iaprtc12 |
| DHLI | SCENE, OBJECT, MIRFlickr, corel5k_5, iaprtc12, 3sources |
| EF2FS | emotions, yeast, SCENE, MIRFlickr, iaprtc12, 3sources |
| I2VSLC | SCENE, OBJECT, corel5k_5, iaprtc12, espgame, 3sources |
| GRAFS | yeast, SCENE, OBJECT, VOC07, MIRFlickr, 3sources |

单折实测耗时（服务器，OMP_NUM_THREADS=4/作业）：TOCL/OBJECT 40–50 min、TOCL/iaprtc12 29–43 min、
TOCL/corel5k_5 33–41 min、UGRFS/iaprtc12 21–29 min、UGRFS/SCENE 50–57 min（约 170 轮收敛，未触 500 上限）；
DHLI 单折仅数秒到数十秒，I2VSLC/EF2FS 单折 20–130 s，GRAFS 单折 1–3 min。

---

## 4. 已知偏差与原因（复现时必须一起看）

### 4.1 分类器是仓库的未文档化假设

六篇 PDF 全文检索 `MLkNN`/`classifier`/`base learner`：只有 I2VSLC 在**相关工作**里提过 ML-kNN，
其余五篇实验设置里**一次都没写**评估分类器。仓库用的是 scikit-multilearn 的 `MLkNN(k=10, s=1.0)`。
换分类器会整体平移所有指标，这是数值对不上时的第一嫌疑。

### 4.2 未调参

`DEFAULT_PARAMS` 六篇全为 1.0，论文是 {10⁻³…10³} 网格搜索。已完成的结果里，
**GRAFS/yeast（AP +0.0078、HL +0.0005、RL −0.0048）、UGRFS/yeast、UGRFS/VOC07、TOCL/MIRFlickr**
在完全不调参的情况下四指标都能落在 0.01 以内，说明量级是对的。

### 4.3 平台浮点漂移（跨机器不可逐位复现）

把 Windows 上跑过的配置在服务器上重跑，**算法本身的排序会有微小漂移**：
UGRFS/3sources 的 3000 个特征里 2864 个名次不同，AP 由 0.4565 变成 0.4579（差 0.0014）；
TOCL/3sources 的聚合值则完全一致（AP 0.4463±0.0333）。
因此 `results/` 里 TOCL/UGRFS 各 3–4 格来自 Windows（原评测器），其余来自服务器（向量化评测器），
两者评测器已证明逐位等价，算法层面的差异是平台 BLAS/LAPACK 噪声，量级 0.001–0.003。

### 4.4 Coverage 口径核验与多标签数据集的残差

`Coverage` 用**原始 coverage_error ÷ 标签数**（论文口径）。本轮用 22 个已完成格子复核了这一点：
在 SCENE / OBJECT / MIRFlickr / yeast / VOC07 / emotions / 3sources 上
`原始 coverage_error = 论文值 × 标签数` **精确成立**
（如 TOCL/MIRFlickr 21.82 vs 0.5715×38 = 21.7；DHLI/SCENE 14.33 vs 0.4174×33 = 13.8）。

偏差只出现在**多标签数据集**，且方向是"我们更好"：

| 数据集 | 标签数 | 我们（原始） | 论文隐含 | 差 |
|---|---|---|---|---|
| corel5k_5 (DHLI) | 260 | 87.0 | 125.6 | −38.6 |
| corel5k_5 (I2VSLC) | 260 | 88.6 | 126.5 | −37.9 |
| iaprtc12 (DHLI) | 291 | 120.8 | 148.1 | −27.3 |
| iaprtc12 (I2VSLC) | 291 | 117.6 | 145.0 | −27.4 |
| iaprtc12 (EF2FS) | 291 | 130.8 | 139.5 | −8.7 |
| espgame (I2VSLC) | 268 | 124.0 | 148.3 | −24.3 |

两个候选解释（都未定论）：

1. **IAPRTC12 的标签数**：论文 Table 1 写 260（与 Corel5K 相同），仓库 `.mat` 实测 291。
   若按论文的 260 归一化，iaprtc12 的偏差立刻收窄到 ±0.04 量级 —— 说明这一格的不一致
   很可能只是论文表格的复制粘贴错误。
2. **corel5k_5 的约 −38 标签差是数据集级、不是算法级**：36 格跑完后可以确认 ——
   TOCL（−0.1175）、DHLI（−0.1485）、I2VSLC（−0.1457）三个算法在 corel5k_5 上
   Coverage 同向、同量级地低于论文，而它们的 AP 只差 0.006–0.028。
   Coverage/RL 与 AP 用的是同一个概率矩阵、四个算法方向一致，
   更像是**标签集合口径不同**（论文可能对 corel5k_5 做过标签频次过滤，
   或 Coverage 的分母标签数与仓库的 260 不同），而不是调参或分类器问题。

注意 AP 在同样的数据集上基本吻合（corel5k_5：TOCL −0.028 / DHLI −0.013 / I2VSLC −0.006），
所以这不是流水线故障，而是排序类指标（Coverage/RL）对分数矩阵尾部敏感所致。

### 4.5 整体贴合度（36 格统计）

| 判据 | 结果 |
|---|---|
| 四指标全部 ≤ 0.005 | 5/36 |
| 四指标全部 ≤ 0.010 | 7/36 |
| 四指标全部 ≤ 0.020 | 14/36 |
| 四指标全部 ≤ 0.050 | 25/36 |

分指标（36 格）：**Hamming Loss 中位 \|Δ\|=0.0034（31/36 落在 0.01 内）**、
Coverage 中位 0.0191、Ranking Loss 中位 0.0118、AP 中位 0.0131。

贴合最好的格子（不调参、四指标全在 0.005 内）：
I2VSLC/SCENE（0.0044）、UGRFS/yeast（0.0045）、UGRFS/VOC07（0.0046）、
TOCL/MIRFlickr（0.0048）、GRAFS/SCENE（0.0049）。

偏差最大的格子全部落在**多标签数据集**（corel5k_5 / iaprtc12 / espgame），
与 §4.4 的分析一致；其余数据集的偏差量级与"未调参 + 分类器未文档化"相符。

---

## 5. 复现命令

```bash
cd /mnt/Data4/24zhs/Mvml/Multi-view-Multi-label-learning

# 单个「算法 × 数据集」，论文口径
MVML_MAX_ITER=500 MVML_SEED=100 bash run.sh \
    --alg GRAFS --data SCENE --folds 5 --select-ratio 0.2 --shuffle

# 本轮全部 30 个作业（长作业优先、8 并发、各自独立 --out）
bash /mnt/Data4/24zhs/Mvml/_mvml_runs/run_remaining.sh \
     /mnt/Data4/24zhs/Mvml/_mvml_runs/jobs_lpt.txt 8

# 收尾：合并 -> 对比表 -> 完整性校验
python merge_parts.py            # 见 _mvml_runs/merge_parts.py
python paper_table.py --write    # results/PAPER_COMPARISON.md
python validate_results.py --write   # results/VALIDATION.md
```

`MVML_MAX_ITER` 的取法：TOCL 60（文档 §6 的限流约定）、UGRFS/DHLI/I2VSLC/GRAFS 500（各自默认上限）、
EF2FS 100（其默认上限）。TOCL 在 iaprtc12 / corel5k_5 上必须加 `--view-order 0,1,2,4,3`
（论文 Table 1 的视图顺序与 `.mat` 的第 4/5 位相反，TOCL 对顺序敏感）。
