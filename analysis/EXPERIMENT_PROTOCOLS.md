# 六篇论文的实验流程对照

> 覆盖 `alg/` 对应的六篇论文：TOCL（ACM MM 2025）、UGRFS（AAAI 2025）、
> EF2FS（Pattern Recognition 2025）、DHLI（AAAI 2024）、I2VSLC（Information Sciences 2024）、
> GRAFS（Information Sciences 2024）。
>

---

## 0. 一句话总结

**六篇用的是同一套实验模板** —— 同一个课题组（Pingting Hao / Huijie Zhang）的系列工作，
实验部分的骨架完全一致：**5 折 CV × 特征百分比 1%–20% × 参数网格 {10⁻³…10³} × 四个指标
（AP/Coverage/HL/RL）**。差异只在：用哪 6 个数据集、跟哪些方法比。

三者是所有论文都验证过的协议默认值：

| 协议项 | 取值 | 六篇是否一致 |
|---|---|---|
| 折数 | 5 | ✅ 六篇全部 |
| 特征子集范围 | 总特征数的 1% – 20% | ✅ 六篇全部 |
| 参数搜索 | {10⁻³, 10⁻², …, 10³}，网格搜索 | ✅ 六篇全部 |
| 指标 | AP↑, Coverage↓, HL↓, RL↓（四项） | ✅ 六篇全部 |
| 报告形式 | mean ± std | ✅ 六篇全部 |

---

## 1. 共用流程（七步）

```
① 数据集准备（多样本多标签，视图已切好）
        ↓
② 5 折交叉验证拆分（训练 / 测试）
        ↓
③ 在【训练集】上跑特征选择模型 → 学到特征权重 w → 全特征降序排序
        ↓
④ 取 top-k 个特征（k 扫描 1% … 20% 的总特征数）
        ↓
⑤ 用这 k 个特征训练/预测多标签分类器
        ↓
⑥ 算 AP / Coverage / HL / RL 四项指标
        ↓
⑦ 对 20 个百分比点取均值，跨 5 折报 mean ± std
```

补充分析（各篇按需做，不属于主流程）：参数敏感性分析、消融实验、收敛性分析、
Friedman 检验、复杂度分析。

### 逐句原文依据

**5 折 + mean±std + 四项指标**

- TOCL：「mean accuracy along with standard deviation obtained from **fivefold cross-validation**.
  The parameters for each method are tuned within the range of 10⁻³, 10⁻², ..., 10³. …
  with **four** widely adopted metrics employed for assessment: Average Precision (AP),
  Coverage, Hamming Loss (HL), and Ranking Loss (RL).」
- UGRFS：「**four** widely adopted metrics … i.e., Average Precision (AP), Coverage,
  Hamming Loss (HL) and Ranking Loss (RL). … mean accuracy along with standard deviation
  obtained from **five-fold cross-validation**. The parameters for each method are tuned
  within the range of {10⁻³, 10⁻², ..., 10³}.」
- DHLI：「On each dataset, **five-fold crossvalidation** is performed and the mean accuracy,
  as well as standard deviation, are reported. The parameters of the regularization paradigm
  are searched in set {10⁻³, 10⁻², ..., 10³}, and four metrics commonly used in this field are
  selected … i.e., Average Precision (AP), Coverage, Hamming Loss (HL) and Ranking Loss (RL).」
- EF2FS：「The experimental results are obtained using a **five-fold cross-validation**, and the
  mean accuracy and standard deviation for each fold are calculated following a grid search
  conducted with the range of 10⁻³, 10⁻², …, 10³.」
- I2VSLC：「… conducting experiments using **five-fold cross-validation**. To determine the optimal
  trade-off parameters, we perform a grid search over the range of 10⁻³, 10⁻², …, 10³.」
- GRAFS：「… are obtained through a **five-fold cross-validation** process, employing a grid search
  within the range of 10⁻³, 10⁻², ⋯, 10³ for parameter tuning. … evaluated using four metrics:
  Average Precision (AP), Coverage, Hamming Loss (HL), and Ranking Loss (RL). … reported as the
  mean value and standard deviation.」

**特征百分比 1%–20%**

- TOCL：「we report experiment results across all datasets under **feature percentages ranging
  from 1% to 20%**.」
- UGRFS：「Each value is calculated based on results obtained using **one to twenty percentage
  features**.」
- DHLI：「Table 2-3 list the performance results of different methods for varying percentage of
  features, with the range of percentages **from one to twenty**.」
- EF2FS：「across varying percentages of features. The range of percentages is set **from 1 to 20**.」
- I2VSLC：「The tables illustrate the results obtained by evaluating these metrics **from 1 to 20
  percentages** of features.」
- GRAFS：「we measure performance using varying numbers of features, ranging **from 1% to 20% of
  the total number of features**.」

**聚合规则（关键，且只有一篇写明）**

- GRAFS：「**The mean value for each metric is calculated by all sample points.**」
  → 即把 1%…20% 这 20 个（或该区间内所有采样）点的指标取**均值**。
- 其余五篇 **只说了扫 1%–20%，没说怎么把 20 个数合成表里的一个数**。
  本仓库 `main.paper_aggregate()` 采用「对 1%…20% 的 20 个百分比点取均值」——
  这个做法现在有 GRAFS 的原文背书，但严格说其余五篇仍属 **【推断】**。

**参数敏感性分析的做法**

- DHLI：「The parameter is **individually tuned while keeping the other parameters fixed**, and the
  grid search is conducted over a predefined range.」
  → 这正是本仓库 `tune.py` 实现的「单参数扫描、其余固定」。
- I2VSLC / EF2FS / GRAFS 也都有对应的 parameter analysis 小节（三维曲面或曲线）。

---

## 2. 逐篇差异

| | TOCL (MM'25) | UGRFS (AAAI'25) | EF2FS (PR'25) | DHLI (AAAI'24) | I2VSLC (InfSci'24) | GRAFS (InfSci'24) |
|---|---|---|---|---|---|---|
| **数据集（各 6 个）** | SCENE, OBJECT, MIRFlickr, Corel5K, IAPRTC12, 3Sources | yeast, SCENE, VOC07, MIRFlickr, IAPRTC12, 3Sources | emotions, yeast, SCENE, MIRFlickr, IAPRTC12, 3Sources | SCENE, OBJECT, MIRFlickr, Corel5K, IAPRTC12, 3Sources | SCENE, OBJECT, Corel5K, IAPRTC12, ESPgame, 3Sources | yeast, SCENE, OBJECT, VOC07, MIRFlickr, 3Sources |
| **对比方法** | MSFS, M2LD, DHLI, EF2FS + MoRE, MRDM, MDFS, CLML（8 个） | M2LD, MSFS + MoRE, MRDM, MIFS, CLML（6 个） | M2LD, MSFS + SSFS, MoRE, MRDM, CLML（6 个） | MSFS, ELSMML, M2LD + MoRE, MRDM, CLML, SCMFS（7 个） | M2LD, MSFS, ELSMML + MoRE, MDFS, CLML, MRDM（7 个） | M2LD, MSFS + SSFS, MoRE, MRDM, CLML（6 个） |
| 额外分析 | Friedman 检验 | 消融（UGRFS_v1/v2/v3） | Friedman 检验 | 消融 | 参数分析（OBJECT/ESPGame） | 消融、收敛性分析 |
| 结果表 | Table 2（AP+Coverage）、Table 3（HL+RL） | Table 2、Table 3 | Table 3、Table 4 | Table 2、Table 3 | Table 3、Table 4 | Table 3、Table 4 |

> 注：各篇的「对比方法」分两类 —— 多视图多标签方法（M2LD/MSFS/ELSMML/DHLI/EF2FS）
> 与多标签特征选择方法（MoRE/MRDM/MDFS/CLML/SSFS/SCFS/MIFS/SCMFS）。
> 本仓库只实现了其中 6 个**被比方法里的**算法（DHLI/EF2FS/GRAFS/TOCL/UGRFS/I2VSLC），
> 其余对比方法（M2LD、MSFS、MoRE、MRDM、CLML 等）**仓库没有实现**。

---

## 3. 与仓库代码的逐步对应

| 论文流程步骤 | 仓库里的实现 | 是否忠实 |
|---|---|---|
| ① 数据准备 | `main.get_data()`；`emotions`/`yeast` 做 3 等频分箱 | 分箱这一点**论文没写**，是上游代码的既有处理 |
| ② 5 折拆分 | `main.split_data_fold()` + `run_one()` 的共享排列 | 论文只说 5 折，没说是否分层/随机；我们用 `--shuffle` 的随机划分 |
| ③ 训练集上学特征权重 | `alg/*.py` 各入口，返回 `record['idx']` 全特征排序 | 一致 |
| ④ 取 top-k | `select_num = int(n_features × --select-ratio)`，扫 k=1..select_num | 仓库扫**每个整数 k**，比论文的 20 个百分比点更密 |
| ⑤ 用选出的特征训练/预测 | `main._make_mlknn()` → scikit-multilearn `MLkNN`，默认 **k=10, s=1.0** | ⚠️ **论文正文从未写明用哪个分类器**，见 §4 |
| ⑥ 四项指标 | `main.evaluate_ranking()` | 一致；但仓库多算了一个 `ZL`（zero-one loss），**论文没有这一项** |
| ⑦ 聚合 + mean±std | `main.paper_aggregate()`、`perfold_*.csv`、`*_paper_mean/std` | 一致（GRAFS 的 `all sample points` 均值口径） |
| 参数网格 | `DEFAULT_PARAMS` 全为 `1.0`；`tune.py` 做单参数扫描 | ⚠️ 默认值不是论文的搜索结果，必须调参才可比 |
| Coverage 口径 | `CV_paper_mean_norm` = `coverage_error / n_labels` | 论文表里的 Coverage 是归一化值（实测确认，见 `REPRODUCE_TOCL_UGRFS.md` §1.3） |

---

## 4. 复现时必须注意的三个坑

### 4.1 六篇论文都没有写明评估用的分类器

我对六篇 PDF 全文检索了 `MLkNN` / `ML-kNN` / `MLKNN` / `kNN` / `KNN` / `classifier` /
`classiﬁer` / `base learner` / `nearest neighbor`：

- **只有 I2VSLC 出现 “ML-kNN”**，而且是在**相关工作**里介绍方法时提到的，
  不是实验设置：「Another notable method, ML-kNN, the actual theory of which is based on the
  k-nearest neighbor algorithm…」
- 其余五篇**一次都没有出现**评估分类器的名字。
- 指标部分只引用了两篇综述（Zhang & Zhou 2013；Gibaja & Ventura 2015），那是指标定义来源。

**结论**：`MLkNN(k=10, s=1.0)` 是从**上游代码**继承的设定，不是论文正文记载的。
它符合多标签特征选择领域的惯例，但严格说这是本仓库的一个**未文档化假设**。
如果复现数值对不上，第一个该怀疑的就是这里（换分类器或换 k 会整体平移所有指标）。

### 4.2 聚合规则只有 GRAFS 写明

只有 GRAFS 明确说「The mean value for each metric is calculated by all sample points」。
其余五篇只说扫了 1%–20%。如果它们实际用的是「取最优」或「取某个固定百分比」，
表里的数就会和我们算的均值有系统性差异。

### 4.3 数据集口径有两处不一致

- **IAPRTC12 的标签数**：TOCL 与 UGRFS 的 Table 1 都写 **260**（与 Corel5K 相同），
  而仓库 `iaprtc12.mat` 实测 **291**。291 是该数据集的通行值，疑似论文表格复制粘贴错误。
- **5 视图数据集的视图顺序**：论文表里是 `DH, DHV3H1, GIST, HHV3H1, HH`，
  仓库 `.mat` 里是 `DH, DHV3H1, GIST, HH, HHV3H1`（第 4/5 位对调）。
  已实测：UGRFS 对顺序基本不敏感，**TOCL 敏感**。
  详见 `REPRODUCE_TOCL_UGRFS.md` §1.4。

---

## 附：重建 PDF 文本缓存

```bash
mkdir -p _pdftext
for f in pre-pdf/*.pdf; do
    pdftotext -q "$f" "_pdftext/$(basename "${f%.pdf}").txt"
done
```

Windows PowerShell：

```powershell
$out = "_pdftext"; New-Item -ItemType Directory -Force -Path $out | Out-Null
Get-ChildItem pre-pdf\*.pdf | ForEach-Object {
    pdftotext -q $_.FullName (Join-Path $out ($_.BaseName + ".txt"))
}
```

`THBFS.txt` 只有 2.4 KB —— 那是一页宣传材料，**没有全文也没有实现代码**，
所以「六篇论文」指上面六篇，不含 THBFS。
