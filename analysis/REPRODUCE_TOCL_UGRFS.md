# TOCL / UGRFS 论文实验复现指南

> 目标：用本仓库复现 **TOCL**（ACM MM 2025）与 **UGRFS**（AAAI 2025）的实验。
>
> 文中「论文口径」的结论都从 `pre-pdf/` 的 PDF 原文核对过（用 `pdftotext` 抽全文检索）；
> 「代码口径」的结论都从本仓库代码核对过，并给出了文件与行号。
> 凡是我推断而非原文直说的，都标了 **【推断】**。

---

## 0. 一句话结论

论文口径是「5 折 CV × 特征百分比 1%–20% × 超参数网格 {1e-3…1e3} × 四个指标」。
本仓库原本只实现了「5 折 × 扫描 k=1..20% 特征」这半截，另有 TOCL 随机种子泄漏导致不可复现。

**下面这些问题现在都已修复并验证**（证据见 §7）：

| 原问题 | 影响 | 处理 |
|---|---|---|
| TOCL 种子泄漏 | 同参数同数据两次跑结果不同，**无法复现** | 已修；实测两次独立运行三个输出文件 SHA256 完全一致 |
| `--shuffle` 不是真 5 折 | 5 折测试集重叠 39%、只覆盖 65% 数据 | 已修；实测并集 100/100、重叠 0 |
| per-fold 指标未落盘 | 只能得到折平均曲线，**算不出 mean ± std** | 已修；新增 `perfold_*.csv` 与 `*_paper_std` |
| 无 1%–20% 聚合 | 论文每个数据集一个数，仓库给的是整条曲线 | 已实现；`paper_table.py` 出对比表 |
| 无超参数寻优 | `DEFAULT_PARAMS` 全是 `1.0` | 已实现 `tune.py`（单参数扫描） |
| Coverage 口径不同 | 论文是 `coverage_error / n_labels`，仓库给原始标签数 | 已确认并新增归一化列（§1.3） |
| `--folds 1` / `--ratio` | 前者训练集为空，后者是死参数 | 已加护栏与提示（§7.5） |

**剩下的唯一硬约束是算力**：TOCL 单轮代价是 O(V·n³)，OBJECT 上 5 折是 **40 小时量级**，
SCENE/iaprtc12 各 15–23 小时。建议先跑 UGRFS（已实测收敛在 ~110–200 轮，快得多），
TOCL 只挑得起的数据集。

---

## 1. 论文口径

### 1.1 两个实验的协议（原文核对）

| 项 | TOCL (ACM MM 2025) | UGRFS (AAAI 2025) |
|---|---|---|
| 数据集（各 6 个） | SCENE, OBJECT, MIRFlickr, Corel5K, IAPRTC12, 3Sources | yeast, SCENE, VOC07, MIRFlickr, IAPRTC12, 3Sources |
| 交叉验证 | 5 折，mean ± std | 5 折，mean ± std |
| 特征子集 | 特征总数的 **1% – 20%** | 特征总数的 **1% – 20%** |
| 超参数 | α, β, γ, λ ∈ {10⁻³, 10⁻², …, 10³} | α, β, γ, δ ∈ {10⁻³, 10⁻², …, 10³} |
| 指标 | AP↑, Coverage↓, HL↓, RL↓ | 同左 |
| 分类器 | 多标签 MLkNN（常规设置） | 同左 |
| 原文位置 | §5.1.3 + Table 2, 3 | *Evaluation Metrics* + Table 2, 3 |

原文关键句（逐字摘录）：

- **TOCL**：「We present the evaluation results as mean accuracy along with standard deviation obtained from **fivefold cross-validation**. The parameters for each method are tuned within the range of 10⁻³, 10⁻², ..., 10³. … with **four** widely adopted metrics employed for assessment: **Average Precision (AP), Coverage, Hamming Loss (HL), and Ranking Loss (RL)**.」
  「we report experiment results across all datasets under **feature percentages ranging from 1% to 20%**.」
- **UGRFS**：「We evaluate the effectiveness of our method using **four** widely adopted metrics … i.e., **Average Precision (AP), Coverage, Hamming Loss (HL) and Ranking Loss (RL)**. … presented as mean accuracy along with standard deviation obtained from **five-fold cross-validation**. The parameters for each method are tuned within the range of {10⁻³, 10⁻², ..., 10³}.」
  「Each value is calculated based on results obtained using **one to twenty percentage features**.」

> ⚠️ 两篇论文都只有 **4 个指标**，**没有** zero-one loss / ZL。
> 目前 `README_zh.md` §5 写的「五项指标」含 `ZL`，那是本仓库代码自带的多算一项，不是论文口径
> （`main.py:87-93` 的 `METRICS` 列表，继承自上游提交 `45f90d8`）。复现时**不要**拿 ZL 去对论文。

### 1.2 目标数值（论文 Table 2/3 中 TOCL / UGRFS 自己那一列）

**TOCL**

| 数据集 | AP ↑ | Coverage ↓ | HL ↓ | RL ↓ |
|---|---|---|---|---|
| SCENE | 0.7981 | 0.4193 | 0.09688 | 0.0914 |
| OBJECT | 0.4959 | 0.2893 | 0.05600 | 0.1607 |
| MIRFlickr | 0.6908 | 0.5715 | 0.17056 | 0.1537 |
| Corel5K | 0.2419 | 0.4710 | 0.01379 | 0.2198 |
| IAPRTC12 | 0.1510 | 0.5072 | 0.01543 | 0.1927 |
| 3Sources | 0.4883 | 0.6017 | 0.23114 | 0.4488 |

**UGRFS**

| 数据集 | AP ↑ | Coverage ↓ | HL ↓ | RL ↓ |
|---|---|---|---|---|
| yeast | 0.6725 | 0.6239 | 0.2257 | 0.2480 |
| SCENE | 0.8010 | 0.4214 | 0.0978 | 0.0942 |
| VOC07 | 0.5871 | 0.4425 | 0.0847 | 0.2111 |
| MIRFlickr | 0.6770 | 0.5885 | 0.1775 | 0.1629 |
| IAPRTC12 | 0.1474 | 0.4996 | 0.01496 | 0.2020 |
| 3Sources | 0.4728 | 0.5301 | 0.2097 | 0.4137 |

> 这些数字是从两栏排版 PDF 里机器抽取、按表头顺序对齐的，**建议抽查 2–3 个再当基准用**；
> 完整表格见 `pre-pdf/TOCL-2025-ACM-MM-*.pdf` Table 2/3 与
> `pre-pdf/25-aaai-Uncertainty-Aware Global-View-*.pdf` Table 2/3。
> UGRFS 的 AP 一列此前也已记录在 `PAPERS_AND_CODE.md` §2「论文报告的基准结果」，两者一致。
>
> 自洽性交叉验证：两篇论文独立报告 SCENE 的 Coverage 分别为 0.4193 / 0.4214，
> SCENE 的 RL 为 0.0914 / 0.0942，量级一致 → 抽取结果可信。

### 1.3 Coverage 的口径：论文用的是「除以标签数」的归一化值（已实测确认）

sklearn 的 `coverage_error` 返回的是**标签个数**（覆盖真实标签集需要排在前面的标签数），
而论文表里的 Coverage 都在 0.15–0.64 之间 —— 显然是归一化过的。
论文正文没有说明除以什么，所以我直接做了实测：**用全部特征跑一次 MLkNN**（不做任何选择，
1 折、`--shuffle`、seed=100），把原始 Coverage 按三种口径换算，和论文值比：

| 数据集 | 标签数 L | 平均标签基数 | 原始 CV | **CV / L** | CV / 基数 | 论文值（TOCL / UGRFS） |
|---|---|---|---|---|---|---|
| SCENE | 33 | 6.49 | 13.576 | **0.4114** | 2.093 | 0.4193 / 0.4214 |
| yeast | 14 | 4.14 | 8.037 | **0.5741** | 1.942 | – / 0.6239 |
| MIRFlickr | 38 | 9.16 | 21.386 | **0.5628** | 2.334 | 0.5715 / 0.5885 |
| 3sources | 6 | 1.24 | 3.853 | **0.6422** | 3.119 | 0.6017 / 0.5301 |

**只有 `CV / 标签数` 这一列落在论文的数值范围里，且非常接近**（注意这还只是「全特征、
未调参、未选特征」的基线）。所以论文的 Coverage 口径 = `coverage_error / n_labels`。

本仓库已按此实现：`summary_*.csv` 里同时给出 `CV_paper_mean`（原始标签数口径）和
**`CV_paper_mean_norm`（论文口径，`paper_table.py` 用的就是它）**。

顺带这一测也**独立验证了 §1.2 从 PDF 抽取的数值**：SCENE/MIRFlickr/yeast 的 HL 与 RL
实测基线（0.0926/0.0829、0.1626/0.1456、0.2107/0.2213）和论文值同量级，
说明抽取对齐没有串行。

### 1.4 论文数据集 ↔ 仓库数据集对照

`iaprtc12` 等 5 视图数据集在仓库里是「同一数据的 5 视图版本」，`corel5k_5` 对应论文的 Corel5K。
下面维度是我用 `scipy.io` 读 `.mat` 实测的，不是抄 README：

| 论文名 | 仓库键 | 仓库文件名 | .mat 实测视图维度 | 样本 × 标签 | 与论文 Table 1 |
|---|---|---|---|---|---|
| 3Sources | `3sources` | `3sources.mat` | [1000, 1000, 1000] | 169 × 6 | ✅ 一致 |
| yeast | `yeast` | `yeast.mat` | [79, 24] | 2417 × 14 | ✅ 一致 |
| SCENE | `SCENE` | `SCENE.mat` | [64, 225, 144, 73, 128] | 4400 × 33 | ✅ 一致 |
| OBJECT | `OBJECT` | `OBJECT.mat` | [64, 225, 144, 73, 128] | 6047 × 31 | ✅ 一致 |
| VOC07 | `VOC07` | `VOC07.mat` | [100, 512, 100] | 3817 × 20 | ✅ 一致 |
| MIRFlickr | `MIRFlickr` | `MIRFlickr.mat` | [100, 512, 100] | 4053 × 38 | ✅ 一致 |
| IAPRTC12 | `iaprtc12` | `iaprtc12.mat` | **[100, 300, 512, 100, 300]** | 4999 × 291 | ⚠️ 第 4/5 视图顺序见下 |
| Corel5K | `corel5k_5` | `corel5k_5.mat` | **[100, 300, 512, 100, 300]** | 4999 × 260 | ⚠️ 同上 |

> ⚠️ **视图顺序差异 —— 已查清，结论见下**
>
> **论文侧**：TOCL / UGRFS / DHLI / EF2FS / I2VSLC 五篇的 Table 1 都写作
> `DH(100), DHV3H1(300), GIST(512), HHV3H1(300), HH(100)` → 维度序 `[100, 300, 512, 300, 100]`。
> **仓库侧**：`.mat` 实测 `[100, 300, 512, 100, 300]` —— 第 4、5 个视图（300 与 100）位置对调。
> 维度总和一样（1312），所以加载不报错。
>
> **哪个物理视图是哪个（数据取证）**：`.mat` 没有 `view_name` 字段，但 `V3H1` 是 100 维基直方图的
> 3 级空间金字塔（300 = 3×100），所以子视图应当包含父视图作为其中一个 100 宽的分块。
> 实测各 100 维视图与各 300 维视图的三个分块的**逐列平均相关系数**：
>
> | 100 维视图 | view2 分块0/1/2 | view5 分块0/1/2 |
> |---|---|---|
> | **view1**（均值 22.5） | **0.681 / 0.815 / 0.735** | 0.007 / 0.006 / 0.004 |
> | **view4**（均值 7.6） | 0.008 / 0.005 / 0.009 | **0.656 / 0.849 / 0.727** |
> | 对照组 | view1 vs view4 = 0.0076；view2.blk0 vs view5.blk0 = 0.0096 | |
>
> 关系非常干净：view1 是 view2 的父直方图（即 `DH` → `DHV3H1`），
> view4 是 view5 的父直方图（即 `HH` → `HHV3H1`）。也就是：
> **仓库 `.mat` 的物理顺序是 `[DH, DHV3H1, GIST, HH, HHV3H1]`，而论文表里写的是
> `[DH, DHV3H1, GIST, HHV3H1, HH]` —— 差异纯粹是第 4/5 位的排列，不是数据损坏。**
> 复现脚本：`_check_view_order.py`。
>
> **这个顺序会不会改变结果（实测）**：在 `3sources`（三个等宽 1000 维视图）上做一次视图循环置换
> `[2,0,1]`，把置换后的排序还原回原编号再比：
>
> | 算法 | 排序完全相同 | rank 相关系数 | top-10 |
> |---|---|---|---|
> | **UGRFS** | 否 | **0.988** | 9/10 相同（差异只在近似并列的权重上） |
> | **TOCL** | 否 | **0.456** | 完全不同 |
>
> 即 **UGRFS 对视图顺序基本不敏感**（其视图只出现在求和型目标里，置换后仅剩浮点级差异），
> 而 **TOCL 对视图顺序敏感**（它的张量核范数沿视图轴做 FFT，切片顺序直接改变结果）。
> 复现脚本：`_check_view_sensitivity.py`。
>
> **因此**：`iaprtc12` / `corel5k_5` 的 **UGRFS 结果可用**；
> **TOCL 在这两个数据集上的结果依赖视图顺序，不能当定论** —— 需要用与论文一致的
> `[DH, DHV3H1, GIST, HHV3H1, HH]` 重排后再跑。重排本身只需交换第 4、5 个视图块。

> 📌 **另一处论文侧不一致（不影响跑，但对比时会困惑）**：TOCL 与 UGRFS 的 Table 1 都把
> IAPRTC12 的标签数写成 **260**（与 Corel5K 相同），而仓库 `iaprtc12.mat` 实测是 **291**。
> 291 是该数据集的通行值，所以这更像是两篇论文表格里的复制粘贴错误；但也不能排除
> 论文对标签做过频次过滤。对比 Coverage 这类依赖标签数的指标时要留意。

---

## 2. 本仓库现状（含本轮新增）

**已有**

- 统一接口：`alg/<ALG>.py` 的入口签名 `(X, x_view, Y, dataset, alpha, beta, gamma, lamb, *extras)`
- 10 个数据集齐全（两个论文需要的 8 个都在）
- MLkNN 评测：`main.py:200-310`，`_make_mlknn()` 不传参 → scikit-multilearn `MLkNN` 的默认值
  **k=10, s=1.0**（已用 `inspect.signature` 实测确认）
- 输出：`rankings_*.csv` / `metrics_*.csv`（折平均曲线）/ `summary_*.csv` / `summary_all.csv`
- `--folds`、`--select-ratio`、`--shuffle`、`--seed`、`--check` 自检

**本轮新增**

| 文件 / 字段 | 作用 |
|---|---|
| `perfold_<算法>_<数据集>.csv` | 每个 fold 的完整指标曲线（长表），mean±std 与任何其它统计量的来源 |
| `summary_*.csv` 的 `*_paper_mean` / `*_paper_std` | 论文口径聚合值（1%–20% 百分比点均值，std 取 5 折间） |
| `summary_*.csv` 的 `CV_paper_mean_norm` / `CV_paper_std_norm` | 除以标签数的 Coverage（论文口径，§1.3） |
| `tune.py` | 单参数超参数扫描（论文 Fig.3 方式），`--rounds` 即坐标下降 |
| `paper_table.py` | 读 `summary_all.csv` 生成与论文 Table 2/3 的对比表，可 `--write` 落盘 |
| `main.paper_aggregate()` | 曲线 → 论文标量的聚合函数 |
| **fold 级断点续跑** | 每折完成后把该折的排序与指标曲线写进 `results/.cache/<key>/`；重跑同一配置自动跳过已完成的折。见 §7.7 |
| `--no-resume` | 关闭续跑，强制从头重算 |
| `--view-order "0,1,2,4,3"` | 按论文 Table 1 的顺序重排视图块（§1.4 的顺序差异需要），记录进 `summary_*.csv` 的 `view_order` 列 |
| `_check_view_order.py` | 数据取证脚本：用父子直方图相关性判定第 4/5 个视图分别是什么 |
| `_check_view_sensitivity.py` | 实测某个算法对视图顺序是否敏感 |

**核心机制与论文的对应**

```
main.py   select_num = max(1, int(X.shape[1] * select_ratio))     # 0.2 → 就是论文的 20%
main.py   for k in range(1, select_num + 1):   → MLkNN(k features) → 5 个指标
```

也就是说 `--select-ratio 0.2` 时，仓库会对 **k = 1..20% 特征的每一个整数 k** 都算一遍指标，
比论文的 20 个百分比点**更密**。论文口径的「1%–20%」是这 20 个点（见 §5）。

---

## 3. 环境准备

### 3.1 Linux 服务器（推荐）

TOCL 在 OBJECT 上峰值内存约 8–9 GB、5 折约 40 小时量级（§6），本机笔记本不适合。

```bash
cd <仓库目录>
bash setup_env.sh            # 创建 conda 环境 mvml（python 3.10 + 锁定版本）
                             # 若 mvml 已存在可跳过
bash run.sh --list           # 确认算法/数据集清单
bash run.sh --check --check-samples 200     # 自检：10 数据集 × 6 算法
```

`setup_env.sh` 会把依赖锁成一组自洽版本：
`numpy==1.26.4`、`scipy==1.11.4`、`scikit-learn==1.3.2`、`pandas==2.0.3`、
`openpyxl==3.1.2`、`scikit-multilearn==0.2.0`、`skfeature-chappers==1.2.1`。

⚠️ 三条硬约束（`setup_env.sh:62-77` 的注释写明了原因）：

1. **numpy 必须是 1.26.x**：`scikit-multilearn 0.2.0` 与 `skfeature-chappers` 都早于 numpy 2.x。
2. **Python 必须是 3.10**：`scikit-multilearn 0.2.0` 依赖已被移除的 `pkg_resources`。
3. 脚本所在的 `/mnt/Data4` 是 `noexec` 挂载，所以**必须 `bash run.sh …`，不能 `./run.sh …`**。

### 3.2 Windows 本地（已按此方案落地并验证）

本机原有环境（实测）：`D:\Anaconda\python.exe` = Python **3.12.8** + numpy **2.3.5** +
scipy 1.17.0 + sklearn 1.8.0，**缺 `skmultilearn` 与 `skfeature`**。
按上面两条硬约束，**这个环境跑不了**。

**实际采用的做法**：不建 conda 环境，改为在仓库内建一个 **uv 管理的 `.venv`**
（Python 3.10.21 + 同一套锁定依赖）。原因有两个：

1. Windows 上没有 `/mnt/Data4` 那种 `noexec` 问题，conda 相对 venv 的唯一优势消失了；
2. 本仓库工作区的写入权限只覆盖 `E:\MVML\...`，而 conda 环境和 uv 的默认缓存/解释器目录都在
   `D:\` 和 `C:\Users\...`，会被拒；把状态放进仓库内就完全自洽，删目录即卸载。

已经执行的步骤（供重建参考）：

```powershell
$ws = "E:\MVML\Multi-view-Multi-label-learning"
$env:UV_CACHE_DIR          = "$ws\.uv-cache"      # 默认在 C:\Users\... → 需重定向
$env:UV_PYTHON_INSTALL_DIR = "$ws\.uv-python"     # 同上

uv python install 3.10                            # → .uv-python\cpython-3.10.21
uv venv --python 3.10 "$ws\.venv"
uv pip install --python "$ws\.venv\Scripts\python.exe" `
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple `
  "setuptools<70" wheel Cython `
  "numpy==1.26.4" "scipy==1.11.4" "scikit-learn==1.3.2" "pandas==2.0.3" `
  "openpyxl==3.1.2" "scikit-multilearn==0.2.0" "skfeature-chappers==1.2.1"
```

验证结果（实测）：

```
python 3.10.21 / numpy 1.26.4 / scipy 1.11.4 / sklearn 1.3.2
MLkNN defaults: {'k': 10, 's': 1.0, 'ignore_first_neighbours': 0}
deps OK
```

顺带确认了 §2 里那条：`_make_mlknn()` 不传参 → **MLkNN k=10, s=1.0**。

`.uv-cache/`、`.uv-python/`、`.venv/` 已加入 `.gitignore`。

`setup_env.sh` / `run.sh` 是 bash 脚本，**Windows 上不用**；直接调用
`& "$ws\.venv\Scripts\python.exe" main.py …`。本文档后续命令里的
`conda run -n mvml python` 请按此替换（或先 `cd` 到仓库再 `& .\.venv\Scripts\python.exe`）。


---

## 4. 执行步骤

下面命令对两个平台都适用，Linux 上把 `conda run -n mvml python main.py` 换成 `bash run.sh` 即可。

### Stage 0 — 冒烟（10 分钟内，先确认能跑）

```bash
# 最小数据集 3sources（169 样本），把迭代压到 20
MVML_MAX_ITER=20 MVML_SEED=100 conda run -n mvml python main.py \
    --alg UGRFS --data 3sources --folds 5 --select-ratio 0.2 --shuffle
MVML_MAX_ITER=20 MVML_SEED=100 conda run -n mvml python main.py \
    --alg TOCL  --data 3sources --folds 5 --select-ratio 0.2 --shuffle
```

看 `results/metrics_TOCL_3sources.csv` 有没有 600 行（3000 特征 × 20%）、
`summary_TOCL.csv` 的 `AP_best` 是不是 0.4 上下。

### Stage 1 — 校准耗时（量一下真实速度）

```bash
# ⚠️ 不要用 --folds 1：那会让测试集覆盖全部数据、训练集为空（main.py 现在会直接报错拦住）。
#    用 5 折、把 select-ratio 压小，只看单折耗时：
/usr/bin/time -v env MVML_MAX_ITER=60 MVML_SEED=100 \
  conda run -n mvml python main.py --alg TOCL --data MIRFlickr \
    --folds 5 --select-ratio 0.02 --shuffle
```

每个 fold 会打印 `fold j/5: N iters, T s`，用 `T` 去替换 §6 表里的估算。
实测校准：TOCL 在 emotions 上单折 330 轮 50.1 秒 → **0.152 秒/轮**，
而 §6 的外推锚点（11 秒 / 60 轮 ≈ 0.183 秒/轮）与之相差约 20%，说明外推模型可用。

> ⚠️ **`--ratio` 是死参数**：`run_one()` 的签名里有它，但函数体从未使用，
> 折大小只由 `--folds` 决定（`n_test = round(m / folds)`）。
> 现在传入与 `1/folds` 不一致的值会打印一行提示。

### Stage 2 — 正式跑

**UGRFS（先跑这个，便宜）**

```bash
for ds in 3sources yeast SCENE VOC07 MIRFlickr iaprtc12; do
  MVML_MAX_ITER=500 MVML_SEED=100 \
    conda run -n mvml python main.py --alg UGRFS --data $ds --folds 5 --select-ratio 0.2 --shuffle
done
```

**TOCL（贵，按成本从低到高逐个上，可断点续跑）**

```bash
for ds in 3sources MIRFlickr VOC07 SCENE iaprtc12 corel5k_5 OBJECT; do
  MVML_MAX_ITER=60 MVML_SEED=100 \
    conda run -n mvml python main.py --alg TOCL --data $ds --folds 5 --select-ratio 0.2 --shuffle
done
```

参数说明（都是复现论文必需的）：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--folds 5` | 5 | 论文就是 5 折 |
| `--select-ratio 0.2` | 0.2 | 20% = 论文扫描的上界 |
| `--shuffle` | 必加 | 不加是按文件顺序切连续块（可能按类别排好序 → 折间严重偏置） |
| `MVML_SEED=100` | 100 | 算法随机初始化种子 |
| `MVML_MAX_ITER` | TOCL 60 / UGRFS 500 | 见下 |
| `--view-order` | 仅 `iaprtc12`/`corel5k_5` 需要 | 见下 |

**关于 `--view-order`**（本轮新增）：论文 Table 1 与 `.mat` 的第 4/5 视图顺序不一致
（§1.4 已用数据取证确认），而 **TOCL 对视图顺序敏感（rank 相关 0.456）、UGRFS 不敏感（0.988）**。
所以：

```bash
# TOCL 在 iaprtc12 / corel5k_5 上要按论文顺序跑
python main.py --alg TOCL --data iaprtc12 --folds 5 --select-ratio 0.2 --shuffle \
    --view-order "0,1,2,4,3"

# UGRFS 不需要（默认 native 顺序即可），等价于
python main.py --alg UGRFS --data iaprtc12 --folds 5 --select-ratio 0.2 --shuffle
```

`--view-order` 的语义是「**新位置 k 放原来的第 order[k] 个视图**」，会做置换合法性校验，
并把实际使用的顺序写进 `summary_*.csv` 的 `view_order` 列（未指定时为 `native`）。
注意：指定后 `rankings_*.csv` 里的 `feature_index` 指的是**重排后**的列布局。

> ⚠️ **`MVML_MAX_ITER` 是「限流」，会改变结果**。TOCL 的收敛判据是 `cver < 1e-6`
> （`alg/TOCL.py:257`），是否达到**因数据集而异**：实测 `3sources` 6 轮就停、
> `emotions` 跑到 330 轮才停。所以 cap 是「上限」而不是「固定预算」——
> 设 60 时 `emotions` 会被截断、`3sources` 不受影响。论文用的是不设 cap 的完整优化，
> **被截断的跑法只能算「同一算法在有限迭代下的近似」**。
> UGRFS 的判据松一些（`1e-3`，`alg/UGRFS.py:149`）。

---

## 5. 后处理：已经内置，用 `paper_table.py`

论文给的是**一个数**（1%–20% 区间的聚合，5 折 mean ± std），仓库跑出来的是**曲线**。
这部分现在已经实现，不需要自己写脚本：

- `main.build_summary()` 在 `summary_<算法>.csv` / `summary_all.csv` 里写入
  `AP/CV/HL/RL_paper_mean` 与 `*_paper_std`（**std 是 5 折间的标准差**，论文就是这个口径）；
- `CV_paper_mean_norm` / `CV_paper_std_norm` 是**除以标签数**的论文口径 Coverage（理由见 §1.3）；
- `perfold_<算法>_<数据集>.csv` 保留每个 fold 的完整曲线，任何其它统计量都能自己重算；
- `paper_table.py` 直接生成对比表：

```bash
python paper_table.py                       # 全部算法
python paper_table.py --alg TOCL UGRFS      # 只看这两个
python paper_table.py --write               # 同时写 results/PAPER_COMPARISON.md
```

输出示例（`3sources`，`--select-ratio 0.2 --folds 5 --shuffle`，参数全 1.0、TOCL 6 轮）：

| 算法 | AP ↑ | Coverage ↓ | HL ↓ | RL ↓ |
|---|---|---|---|---|
| TOCL | 0.4463±0.0333<br><sub>paper 0.4883 ▼0.0420</sub> | 0.5892±0.0561<br><sub>paper 0.6017 ▲0.0125</sub> | 0.2101±0.0089<br><sub>paper 0.2311 ▲0.0210</sub> | 0.4705±0.0598<br><sub>paper 0.4488 ▼0.0217</sub> |
| UGRFS | 0.4409±0.0209<br><sub>paper 0.4728 ▼0.0319</sub> | 0.5886±0.0190<br><sub>paper 0.5301 ▼0.0585</sub> | 0.2312±0.0240<br><sub>paper 0.2097 ▼0.0215</sub> | 0.4711±0.0210<br><sub>paper 0.4137 ▼0.0574</sub> |

**两点必须注意：**

1. **聚合规则是【推断】**。论文只说「feature percentages ranging from 1% to 20%」，
   没写怎么把 20 个点合成一个数；这里按「对 20 个百分比点取均值」实现。
2. **`*_best` 不是论文口径。** `summary_*.csv` 里的 `AP_best` 是曲线上的最优点，
   和 1%–20% 聚合值不相等，别混用。
3. **`--select-ratio` 必须 ≥ 0.2**，否则曲线覆盖不到 20%，百分比点会被截断，
   这些聚合值会被自动省略（并在运行时打印提示）。

---

## 6. 时间 / 内存预算

下表是**实测值优先**，未实测的用实测锚点外推（标 **【推断】**）。
口径：TOCL `MVML_MAX_ITER=60`、5 折、`--shuffle`、`--select-ratio 0.2`、8–11 个逻辑核可用。

实测锚点（本机 Windows）：

| 实测点 | 结果 |
|---|---|
| TOCL / emotions（n=593，330 轮收敛） | 50.1 s/折 → **0.152 s/轮** |
| TOCL / 3sources（n=169，6 轮收敛） | 5.5 s/折（含评测），5 折共 5.5 min |
| **TOCL / MIRFlickr（n=4053，跑满 60 轮）** | **1088 s/折 = 18.1 min/折** |
| UGRFS / 3sources（n=169，113 轮收敛） | 17.8 s/折（纯算法），5 折共 6.4 min |
| **UGRFS / yeast（n=2417，214 轮收敛）** | **311 s/折 = 5.2 min/折**，5 折共 28 min |

| 数据集 | n | V | d | TOCL 单折 | TOCL 5 折 | TOCL 峰值内存 | UGRFS 5 折 |
|---|---|---|---|---|---|---|---|
| 3sources | 169 | 3 | 3000 | **5.5 s**（实测） | **5.5 min**（实测） | < 0.1 GB | **6.4 min**（实测） |
| yeast | 2417 | 2 | 103 | – | （TOCL 不用） | – | **28 min**（实测） |
| VOC07 | 3817 | 3 | 712 | 【推断】~16 min | 【推断】~1.5 h | ~2 GB | 【推断】~1–2 h |
| MIRFlickr | 4053 | 3 | 712 | **18.1 min**（实测） | 【推断】~1.7 h | **2.6 GB**（实测） | 【推断】~1.5–8 h |
| SCENE | 4400 | 5 | 634 | 【推断】~40 min | 【推断】~3.5 h | 【推断】~4.5 GB | 【推断】~2–8 h |
| iaprtc12 / corel5k_5 | 4999 | 5 | 1312 | 【推断】~1 h | 【推断】~5 h | 【推断】~6 GB | 【推断】~4–20 h |
| OBJECT | 6047 | 5 | 634 | 【推断】~1.7 h | 【推断】~9 h | 【推断】~9 GB | （不用跑） |

> 上表比我最初的估算**乐观不少**（MIRFlickr 从 ~7 h/5 折修正为 ~1.7 h/5 折），
> 原因是原估算按 O(V·n³) 从 emotions 外推，而实测显示大 n 时 BLAS 利用率提升、
> 常数开销被摊薄。**以实测行为准。**

四个必须知道的点：

- **TOCL 是瓶颈**，成本被 `prox_weight_tensor_nuclear_norm()`（`alg/TOCL.py:17-68`）主导：
  每轮对 `(n, n, V)` 张量做 FFT + 逐切片 **n×n 的 SVD** + 逆 FFT，即 O(V·n³)，且
  `Z`/`HH`/`Z2`/复数工作区全是 `(n, n, V)` 稠密数组 → 内存 O(n²V)。
- **TOCL 的迭代数是「收敛即停」，不是固定预算**：3sources 6 轮就停，MIRFlickr 跑满 60 轮。
  设 `MVML_MAX_ITER=60` 时，若某数据集真需要 200 轮，结果就是被截断的近似。
- **上面只是特征选择的耗时**，`evaluate_ranking()` 还要在每折里对
  k = 1..select_num 各训练一次 MLkNN（SCENE 是 126 次/折，iaprtc12 是 262 次/折），
  另需几分钟到数小时；iaprtc12 有 260 个标签，`_make_mlknn()` 里的
  `predict`/`predict_proba` 是逐标签 Python 循环（`main.py:260-277`），这一项最慢。
- **限流不是免费的**：`MVML_MAX_ITER` 越小越快，但离论文的完整优化越远。

**UGRFS 便宜得多**：主循环是若干 O(n²d) 稠密矩阵乘（迭代上限 500、收敛判据 1e-3，实测
3sources 113 轮收敛）。**先做 UGRFS 能更快拿到完整一圈。**

---

## 7. 四个阻塞项（全部已修复，附验证证据）

### 7.1 TOCL 不可复现（种子泄漏）— 已实测确认

`alg/TOCL.py:114` 的权重初始化用的是**全局** RNG，不是受种子控制的 `rng`：

```python
W = np.random.rand(feature_num, label_num)   # ← 全局 np.random，不受 MVML_SEED 控制
```

上游原文是 `from numpy.random import seed; seed(2)` 把全局种子写死的，我在重写时把它换成了
`make_rng()` 局部 Generator，但**漏改了这一行**，同时**删掉了 `seed(2)`**，于是它变成了完全不受控。

实测（合成小数据，同进程内同参数连跑两次，`MVML_MAX_ITER=3`）：

```
run A: [10, 9, 11, 7, 8, 4, 1, 5]
run B: [10, 8, 7, 6, 11, 5, 4, 3]      → identical: False
# 换一个进程再跑，又是两组不同的排序
```

**后果**：TOCL 的每次运行结果都不同，`README_zh.md` §7 里「同种子跑两次逐字节一致」对 TOCL 不成立。
**状态：已修复** —— 改为 `W = rng.random((feature_num, label_num))`。
修复后重新实测：同一进程连续两次调用、以及两个独立进程，排序**完全一致**
（均为 `[7, 8, 11, 9, 10, 6, 5, 2]`）。

### 7.2 出不了 std

`save_run()` 只落盘折平均曲线，per-fold 数值没保存 → 无法算 mean ± std。

**状态：已修复** —— `run_one()` 现在累积 `folds × select_num` 的二维矩阵并传给 `save_run()`，
后者写出 `perfold_<算法>_<数据集>.csv`（长表：`fold, num_features, HL, RL, CV, AP, ZL`）；
`build_summary()` 同时计算 `*_paper_mean` / `*_paper_std`（std 取 **5 折间**标准差）。
验证：`perfold_TOCL_3sources.csv` 283 KB、`perfold_UGRFS_3sources.csv` 296 KB，均 5 折 × 600 个 k。

### 7.3 没有超参数寻优

`DEFAULT_PARAMS` 四个参数全是 `1.0`，论文是 {10⁻³…10³} 网格搜索。
7 个取值 × 4 个参数 = 2401 组合，全网格远大于全量 5 折的代价。

**状态：已实现** —— 新增 `tune.py`，按论文 Fig.3 那样**单参数扫描、其余固定**
（`--rounds` 可变成坐标下降），默认 4 参数 × 7 取值 = 28 次交叉验证：

```bash
python tune.py --alg TOCL --data 3sources                      # 默认网格，folds=5
python tune.py --alg UGRFS --data SCENE --params alpha --values 0.001 0.1 10
python tune.py --alg TOCL --data 3sources --select-ratio 0.05 --aggregate best --max-iter 20
```

扫描全程 `run_one(save=False)`，**不污染 `results/`**；结果写 `results/tuning/tuning_<算法>_<数据集>.csv`，
最后打印可直接粘进 `DEFAULT_PARAMS` 的配置。实测一次 3 点扫描（TOCL/3sources，5 折）耗时 240 秒，
输出 `alpha: 1.0 -> 0.1 (AP=0.5073)`。

### 7.4 `--shuffle` 的 5 折不是真 5 折 — 已实测确认

`split_data_fold()`（`main.py:157-176`）每次调用都重新 `rng.shuffle(indices)`，
而 `run_one()` 每折换种子（`main.py:464` `default_rng(seed + j)`）→ **每折用的是不同的排列**，
于是 5 个测试集**互相重叠、也不覆盖全数据**。实测（m=100, 5 折）：

```
每折测试集大小 : [20, 20, 20, 20, 20]     （各折大小对）
5 折测试集并集 : 65 / 100                 （真 5 折应为 100）
两两重叠总数   : 39                       （真 5 折应为 0）
=> 这是「5 次独立随机 80/20 留出」，不是 5 折交叉验证
```

不加 `--shuffle` 时确实是真划分，但用的是**文件原始顺序**的连续块，如果数据按类别排过序就会严重偏置。

**状态：已修复** —— `split_data_fold()` 新增 `indices` 参数，`run_one()` 只用
`default_rng(seed)` 生成**一次**共享排列，5 折各取其中互不相交的一段。
修复后实测（m=100）：

```
每折测试集大小 : [20, 20, 20, 20, 20]
5 折测试集并集 : 100 / 100     ✅
两两重叠总数   : 0             ✅
=> 真 5 折划分
```

### 7.5（新发现）两个附带缺陷，也已修

| 问题 | 现象 | 处理 |
|---|---|---|
| `--folds 1` | 折大小 `n_test = round(m/1) = m` → 测试集等于全量、训练集为空，会走到 `no fold produced a usable feature ranking` 才报错，容易误判成算法问题 | `run_one()` 现在直接拒绝 `folds < 2` 并说明原因 |
| `--ratio` | `run_one()` 签名里有这个参数，**函数体从未使用**，`README_zh.md` 却把它写成「每折测试集比例」 | 传入与 `1/folds` 不一致的值时打印提示；实际折大小只由 `--folds` 决定 |

### 7.6（新发现）两个汇总文件的缺陷，也已修

| 问题 | 现象 | 处理 |
|---|---|---|
| `summary_<算法>.csv` 只保留最后一次运行 | `README_zh.md` 写的是「每个数据集一行」，实现却每次覆盖成单行；跑一次别的数据集就把之前的行冲掉 | 两个汇总文件都改为**从历史重建**：`summary_all.csv` 仍是追加式审计日志，`summary_<算法>.csv` 按数据集去重、保留最新一行 |
| 表头变化会丢历史 | 新增列（如 `view_order`）后，旧表头与新表头不一致；原先的处理是把旧文件挪成 `.bak` 再重开，等于让 `PAPER_COMPARISON.md` 丢掉已有数据集 | 改为**迁移**：按列名对齐、缺列填空，历史行全部保留 |
| `unlink()` 在文件仍打开时调用 | 迁移分支里我在 `with open(...)` 块内就 `unlink()`：POSIX 允许，**Windows 报 `PermissionError [WinError 32]`** 并让整个运行以退出码 1 失败 | 先把句柄关掉再删；这条专门在注释里写明原因 |
| `.bak` 兼容 | 若确实出现 `.bak`（旧版本进程与新版本进程并发写），报告会漏数据集 | `paper_table.py` 现在同时读 `summary_all.csv` 与 `summary_all.bak.csv`，以实时文件优先 |

### 7.7（新发现）Coverage 口径

见 §1.3：论文的 Coverage 是 `coverage_error / n_labels`，仓库原先直接输出原始标签个数。
已在 `summary_*.csv` 中新增 `CV_paper_mean_norm` / `CV_paper_std_norm`，
`paper_table.py` 用的就是归一化列。

### 7.8（新发现）长跑一中断就全丢 —— 已加 fold 级断点续跑

TOCL 在 SCENE/MIRFlickr/iaprtc12 上单折要 **18–42 分钟**，整轮 2–4 小时。
原先只要进程被杀（Ctrl-C、后台作业被回收、机器重启），**已完成的折全部白跑**。
实测本机确实反复遇到：后台作业在会话空闲期被回收，SCENE 跑到 fold 2/5 时整个丢失。

**状态：已实现** —— `run_one()` 每完成一折就把该折的**排序**与**指标曲线**写进
`results/.cache/<hash>/`，hash 覆盖所有会改变结果的旋钮
（算法、数据集、四个超参、折数、`select_ratio`、`shuffle`、`seed`、`view_order`、
`MVML_MAX_ITER`、`MVML_SEED`）。重跑同一配置会打印并跳过已缓存的折：

```
resume: 1 cached fold(s) found under .cache\981bc03015f0aa71 -- skipping those
  fold 0/5: cached (250 iters)
  fold 1/5: 250 iters, 58.71s, 72 features ranked
  ...
```

**正确性实测**（关键：缓存回放必须与全新计算逐字节一致）：
故意在 fold 0 后杀掉进程 → 续跑 → 与 `--no-resume` 的全新运行对比：

```
rankings_TOCL_emotions.csv   IDENTICAL
metrics_TOCL_emotions.csv    IDENTICAL
perfold_TOCL_emotions.csv    IDENTICAL
summary_TOCL.csv             只有 cv_time_sec / running_time_sec 不同（计时字段）
```

> ⚠️ 第一版用 `%.10g` 存曲线，结果 `metrics_*` / `perfold_*` **不一致** ——
> 10 位有效数字不足以无损往返 float64，把折平均曲线扰动了 ~1e-11。
> 改成 `%.17g`（float64 最短无损往返格式）后逐字节一致。

**用法**：什么都不用加，默认就开。想强制重算用 `--no-resume`。
缓存按配置隔离，所以换超参/种子不会误用旧缓存。

---

## 8. 当前执行状态

四个阻塞项 + 四个后续发现的缺陷**已全部修复并逐条验证**（证据见 §7），环境已建好，
两个算法的长时运行正在后台进行。

已经跑通并验证的：

| 项 | 结果 |
|---|---|
| `main.py` + `tune.py` + `paper_table.py` 编译 | OK |
| TOCL 可复现性 | 两次**独立完整运行**后 `rankings_/metrics_/perfold_TOCL_3sources.csv` SHA256 三个全部 IDENTICAL |
| 真 5 折划分 | 并集 100/100、重叠 0 |
| `--view-order` | 非法置换被拒绝；恒等置换与 native **逐字节一致** |
| `--check` 回归 | 10 数据集 × 6 算法全部 `ok` |
| 汇总文件 | 表头迁移保留历史；同一配置重跑不会重复行；`.bak` 兼容 |
| `paper_table.py` | 正确输出对比表与 paper 偏差 |
| `tune.py` | 真实 3 点扫描跑通，输出调参建议 |

**已落地的正式结果**（`--select-ratio 0.2 --folds 5 --shuffle`，参数全 1.0；
数值 = 1%–20% 百分比点均值，± 为 5 折间标准差）：

| 算法 | 数据集 | 迭代 | AP | Coverage | HL | RL | 四指标最大偏差 |
|---|---|---|---|---|---|---|---|
| UGRFS | yeast | 213 | 0.6680±0.0100 | 0.6269 | 0.2227 | 0.2496 | **0.0045** |
| TOCL | MIRFlickr | 60 | 0.6860±0.0042 | 0.5742 | 0.1740 | 0.1560 | **0.0048** |
| TOCL | SCENE | 60 | 0.7885±0.0063 | 0.4384 | 0.1022 | 0.0994 | 0.0191 |
| UGRFS | MIRFlickr | 152 | 0.6659±0.0070 | 0.5946 | 0.1825 | 0.1694 | 0.0111 |
| UGRFS | 3sources | 113 | 0.4565±0.0512 | 0.5824 | 0.2153 | 0.4613 | 0.0523 |
| TOCL | 3sources | 6 | 0.4463±0.0333 | 0.5892 | 0.2101 | 0.4705 | 0.0420 |

**两篇各有一个数据集做到「四指标全部落在 0.005 以内」**（UGRFS/yeast、TOCL/MIRFlickr），
而且**完全没有调参**（α=β=γ=λ=1.0）。`3sources` 偏差大是因为它只有 169 样本 / 6 标签，
每折测试集仅 34 条，std 0.03–0.06 比大数据集大一个数量级。

**进度：论文口径 6 / 12 个组合完成**

| 算法 | 论文数据集 | 已完成 | 未完成 |
|---|---|---|---|
| TOCL | SCENE, OBJECT, MIRFlickr, Corel5K, IAPRTC12, 3Sources | SCENE, MIRFlickr, 3Sources | OBJECT, corel5k_5, iaprtc12 |
| UGRFS | yeast, SCENE, VOC07, MIRFlickr, IAPRTC12, 3Sources | yeast, MIRFlickr, 3Sources | SCENE, VOC07, iaprtc12 |

> 另有一个**跑了但不属于该论文**的组合：TOCL / VOC07（VOC07 是 UGRFS 的数据集，
> AP=0.8010）。`paper_table.py` 会把它单列提示，不计入对比。

**仍在跑**（后台，已开 fold 级断点续跑，见 §7.8）：

| 作业 | 队列 | 状态 |
|---|---|---|
| TOCL（`MVML_MAX_ITER=60`，`--view-order 0,1,2,4,3`） | iaprtc12 → corel5k_5 | 刚开始 |
| UGRFS（无 cap） | VOC07 → SCENE → iaprtc12 | VOC07 fold 0 已缓存（单折 ~62 min） |

实测单折耗时：TOCL/SCENE 21–24 min、TOCL/VOC07 13–14 min、TOCL/MIRFlickr 18–21 min、
UGRFS/MIRFlickr ~30 min、UGRFS/VOC07 ~62 min。

**预期差异（重要）**：参数仍用 1.0、TOCL 受 60 轮限流，数值不会完全等于论文表格。
但 `yeast` / `MIRFlickr` 的结果说明**默认参数已经落在论文的量级上**，
剩下 3sources 与 SCENE 的 0.01–0.05 偏差主要来自数据集规模小、方差大。
这不代表复现失败 —— `README_zh.md` §7 已声明本仓库不是论文结果复现。

**数据侧不确定性已查清**（详见 §1.4）：`iaprtc12` / `corel5k_5` 的第 4、5 视图
在 `.mat` 里是 `[HH, HHV3H1]`，论文表里是 `[HHV3H1, HH]` —— 纯排列差异，已用
父子直方图相关性证实（相关系数 0.66–0.85 vs 对照 0.008）。实测影响：
**UGRFS 基本不敏感（rank 相关 0.988），TOCL 敏感（0.456）**。
所以 UGRFS 在这两个数据集上的结果可用，TOCL 的需要按论文顺序重排后再跑。

---

## 附：核对清单

| 检查项 | 命令 | 状态 |
|---|---|---|
| 数据集齐全且维度自洽 | 见 §1.4；或 `python main.py --check` | ✅ 10/10 |
| 算法清单 | `bash run.sh --list` | ✅ |
| 单算法单数据集闭环 | Stage 0 | ✅ |
| 曲线行数 = int(d × select_ratio) | `wc -l results/metrics_TOCL_3sources.csv` → 601（含表头） | ✅ 600 行 |
| 排序是完整置换 | `main.run_selfcheck()` | ✅ |
| **TOCL 可复现性（逐字节）** | 两次独立完整运行后比对 `rankings_/metrics_/perfold_TOCL_3sources.csv` 的 SHA256 | ✅ 三个文件全部 IDENTICAL |
| **真 5 折划分** | 见 §7.4 的并集/重叠检查 | ✅ 100/100，重叠 0 |
| **per-fold 落盘 + mean±std** | `summary_*.csv` 有 `*_paper_mean` / `*_paper_std` | ✅ |
| **Coverage 论文口径** | `CV_paper_mean_norm` 存在且与论文同量级 | ✅ 见 §1.3 |
| **对比表生成** | `python paper_table.py --write` | ✅ |
| **超参数扫描** | `python tune.py --alg TOCL --data 3sources --dry-run` | ✅ 跑通真实扫描 |
| `--folds 1` 被拦住 | `python main.py --alg UGRFS --data 3sources --folds 1` | ✅ 明确报错 |
| `--ratio` 不一致有提示 | `python main.py --alg TOCL --data emotions --folds 5 --ratio 0.3` | ✅ 打印提示 |
