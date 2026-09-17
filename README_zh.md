# 多视图多标签特征选择（中文说明）

> 本文档是补充说明，英文原版 `README.md` 保持原样未改动。
> 本仓库上游为东北师范大学 Pingting Hao 课题组的公开实现，当前版本在此基础上
> 修复了若干导致无法运行 / 静默出错的缺陷，**全部修复细节见 [`FIXES.md`](FIXES.md)**。
> 本文档只说明：怎么装、怎么跑、参数含义、输出格式、已做的验证。

---

## 1. 快速开始

> ⚠️ 本仓库位于 `/mnt/Data4`，该分区以 **`noexec`** 挂载，因此
> **不能直接 `./run.sh`**，必须用 `bash run.sh`（原因见第 6 节）。

```bash
bash setup_env.sh          # 创建 conda 环境 mvml 并安装依赖
bash run.sh --list         # 查看可用数据集与算法
bash run.sh --check        # 自检：10 数据集 × 6 算法全部试跑一遍

bash run.sh --alg DHLI --data emotions                     # 单算法单数据集
bash run.sh --alg TOCL --data emotions yeast --shuffle      # 多数据集
bash run.sh --alg all --data all                            # 全网格（很慢）
```

运行时依赖装在名为 `mvml` 的 conda 环境里，`run.sh` 会自动定位并调用它；
可用 `CONDA_ENV=其它环境名 bash run.sh …` 指定别的环境。

## 2. 目录结构

```
main.py              评测主程序：加载数据 → 特征排序 → 取前 k 个 → MLkNN → 5 项指标
alg/
  _util.py           公共工具：随机种子、迭代上限、索引类型、稀疏转稠密、视图维度
  DHLI.py            双层级混合标签识别                     (AAAI 2024)
  TOCL.py            张量对立互补学习                       (ACM MM 2025)
  EF2FS.py           嵌入式特征融合                         (Pattern Recognition 2025)
  GRAFS.py           锚点引导全局视图重构                   (Information Sciences 2024)
  I2VSLC.py          视图特有标签关系                       (Information Sciences 2024)
  UGRFS.py           不确定度感知全局视图重构               (AAAI 2025)
  THBFS_code_description.md   仅流程说明，无实现代码
data/*.mat           10 个基准数据集（约 134 MB）
pre-pdf/             六篇论文全文（TOCL/UGRFS/EF2FS/DHLI/I2VSLC/GRAFS）
                     + THBFS 宣传材料；另含 README.md 与 paper-notes/
results/             运行输出（CSV，首次运行自动创建）
README.md            英文原版说明（未改动）
README_zh.md         本文档
FIXES.md             修复清单与证据
PAPERS_AND_CODE.md   论文与代码对照简介　← 查看各算法对应哪篇论文、变量如何对应
setup_env.sh         环境安装脚本
run.sh               运行包装脚本
_smoke_tocl.py       TOCL 单独冒烟测试（TOCL 太慢时用）
```

## 3. 统一的算法接口

所有算法共用同一个签名，这是它们可以互换调用的基础：

```python
record, n_iter = entry(X, view_dims, Y, dataset, alpha, beta, gamma, lamb [, 额外参数])

record['idx']            # 全部特征编号，按重要性从高到低排序
record['param']          # 超参数记录
record['obj_value']      # 每轮目标函数值
record['running_time']   # 运行耗时
```

- `X`：**所有视图按列拼接**后的矩阵
- `view_dims`：各视图的特征数，**按顺序**消费
  （视图 0 = 前 `view_dims[0]` 列，视图 1 = 接着 `view_dims[1]` 列……）
- `EF2FS` 额外需要 `V_dim`（隐空间维数）、`GRAFS` 额外需要 `kk`（锚点数），
  在 `main.py` 的 `EXTRA_ARGS` 中声明

6 个算法共有的建模思路：随机初始化 + 乘性更新交替优化 + `‖·‖₂,₁` 行稀疏正则，
最后按行 2-范数排序全部特征。

## 4. 命令行参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--alg` | `DHLI` | 算法名，或 `all` |
| `--data` | `emotions` | 一个或多个数据集名，或 `all` |
| `--folds` | `5` | 交叉验证折数 |
| `--ratio` | `0.2` | 每折测试集比例 |
| `--select-ratio` | `0.2` | 评测时扫描的特征比例（共扫 `特征数 × 该比例` 个 k 值） |
| `--shuffle` | 关闭 | 折分前是否打乱（X 与 Y 始终同步打乱） |
| `--seed` | `100` | 折分随机种子 |
| `--out` | `results/` | 输出目录 |
| `--check` | – | 自检模式：每个数据集上试跑全部算法 |
| `--check-samples` | `200` | 自检时每个数据集取样条数 |

环境变量：

| 变量 | 默认 | 含义 |
|---|---|---|
| `MVML_SEED` | 各算法内置（`100`） | 算法的随机初始化种子 |
| `MVML_MAX_ITER` | 各算法内置（500，`EF2FS` 为 100） | 迭代上限。**TOCL 每轮要做一次张量 SVD，是本仓库最慢的算法**，快速试跑建议 `MVML_MAX_ITER=60` |
| `MVML_SELECT_RATIO` | `0.2` | `--select-ratio` 的默认值 |
| `CONDA_ENV` | `mvml` | 供 `run.sh` / `setup_env.sh` 指定 conda 环境名 |
| `CONDA_BASE` | 自动探测 | conda 安装根目录（如 `~/miniconda3`） |

## 5. 输出文件与指标

| 文件 | 内容 |
|---|---|
| `rankings_<算法>_<数据集>.csv` | `rank, feature_index, view`，全部特征的排序及所属视图 |
| `metrics_<算法>_<数据集>.csv` | 各指标随「选取特征数 k」变化的曲线 |
| `summary_<算法>.csv` | 每个数据集一行：各指标的 `best` / `best_k` / `auc` + 超参数 |
| `summary_all.csv` | 所有运行追加汇总，便于跨算法对比 |

五项指标：

| 简写 | 指标 | 方向 |
|---|---|---|
| `HL` | hamming loss | 越低越好 |
| `RL` | label ranking loss | 越低越好 |
| `CV` | coverage error | 越低越好 |
| `AP` | average precision | **越高越好** |
| `ZL` | zero-one loss | 越低越好 |

## 6. 数据集

`emotions`、`yeast`、`3sources`、`SCENE`、`OBJECT`、`VOC07`、`MIRFlickr`、
`corel5k_5`、`iaprtc12`、`espgame`

| 数据集 | 样本数 | 视图数 × 维度 | 标签数 |
|---|---|---|---|
| emotions | 593 | 2（8, 64） | 6 |
| yeast | 2417 | 2（79, 24） | 14 |
| 3sources | 169 | 3 × 1000 | 6 |
| MIRFlickr | 4053 | 3（100, 512, 100） | 38 |
| VOC07 | 3817 | 3（100, 512, 100） | 20 |
| SCENE | 4400 | 5（64, 225, 144, 73, 128） | 33 |
| OBJECT | 6047 | 5（64, 225, 144, 73, 128） | 31 |
| corel5k_5 / iaprtc12 / espgame | 4999 | 5（100, 300, 512, 100, 300） | 260 / 291 / 268 |

说明：`emotions` 与 `yeast` 会先做 3 等频分箱离散化（与原流程一致）；
`corel5k_5 / iaprtc12 / espgame` 是同一数据的 5 视图版本。

### 本机三个环境坑（非代码问题）

1. **`/mnt/Data4` 以 `noexec` 挂载**。若把虚拟环境放在仓库内，其 `.so` 无法被
   mmap，`import numpy` 会报
   `failed to map segment from shared object`。因此本项目改用 **conda 环境**
   （名为 `mvml`，建在 conda 的 envs 目录 `~/miniconda3/envs/mvml`，
   位于可执行的家目录分区）。
   同样原因，脚本不能 `./run.sh` 直接执行，要写 `bash run.sh`。
2. **PyPI 直连不通**（清华镜像可用），且非交互 shell 里
   `conda activate` / `source .venv/bin/activate` 不一定生效。所以
   `run.sh` 直接调用环境内的解释器，`setup_env.sh` 用 `$PY -m pip` 安装，
   都不依赖 `activate`。
3. **`~/.local` 里的包会"骗过" pip**。本机在
   `~/.local/lib/python3.10/site-packages` 下已有 numpy/scipy/scikit-learn/
   scikit-multilearn/skfeature。pip 判定"依赖已满足"时会去看那里，于是
   **跳过安装**，环境里只剩一个 `.dist-info` 而没有真正的代码
   （表现为 `import joblib`、`import skmultilearn` 失败）。因此
   `setup_env.sh` 使用 `--ignore-installed` 强制把文件装进环境；
   `run.sh` 运行时设 `PYTHONNOUSERSITE=1`，避免旧版本遮蔽环境中锁定的版本。
   `setup_env.sh` 末尾会以同样的开关做校验，并断言 numpy/scipy/sklearn/pandas
   确实来自环境目录。

依赖版本锁定为一组自洽组合：`numpy==1.26.4`、`scipy==1.11.4`、
`scikit-learn==1.3.2`、`pandas==2.0.3`、`openpyxl==3.1.2`、
`scikit-multilearn==0.2.0`、`skfeature-chappers==1.2.1`。
**numpy 2.x 不可用**：`scikit-multilearn` 0.2.0 与 `skfeature-chappers` 均早于它。
**Python 3.13 也不可用**：`scikit-multilearn` 0.2.0 依赖已被移除的
`pkg_resources`，故环境固定为 Python 3.10。

## 7. 已做的验证

| 检查项 | 结果 |
|---|---|
| `--check`：10 数据集 × 6 算法（60 组） | 全部 `ok`，且返回排序都是全部特征索引的完整置换 |
| 6 个算法各跑通完整 5 折流程（`emotions`） | 全部退出码 0 并写出 CSV |
| 同种子跑两次 | `rankings_*.csv` 与 `metrics_*.csv` **逐字节一致** |
| 数据加载 `sum(view_dims) == X.shape[1]` | 10 个数据集全部成立 |

参考运行（`emotions`，5 折，`--select-ratio 0.1`，`MVML_SEED=100`），
取扫描范围内各指标的最优值：

| 算法 | 迭代次数 | HL | RL | CV | AP | ZL |
|---|---|---|---|---|---|---|
| DHLI | 37 | 0.3025 | 0.3974 | 3.923 | 0.5978 | 0.928 |
| EF2FS | 100 | 0.2552 | 0.2970 | 3.414 | 0.6761 | 0.814 |
| GRAFS | 204 | 0.2799 | 0.3564 | 3.786 | 0.6221 | 0.868 |
| I2VSLC | 14 | 0.2808 | 0.3462 | 3.672 | 0.6317 | 0.877 |
| TOCL | 60（限流） | 0.2949 | 0.3907 | 3.888 | 0.5932 | 0.936 |
| UGRFS | 173 | 0.2656 | 0.2845 | 3.351 | 0.6826 | 0.816 |

> ⚠️ **注意**
> 1. 上表只是「能跑通」的合理性检查，**不是论文结果复现** —— 超参数全部用的是
>    `main.py` 里的默认值 `1.0`，没有做任何调参。
> 2. 本次修复中包含数个**会改变数值结果**的更正（尤其 `DHLI` 的视图互斥项、
>    `EF2FS` 的图拉普拉斯、以及折内排序的使用方式）。**上游旧代码跑出的结果与
>    本版本不可直接比较，必须重跑。** 详见 `FIXES.md`。
> 3. **THBFS 目前只有流程说明和论文 PDF，没有实现代码。** 若要接入，需补
>    `alg/THBFS.py` 并遵守第 3 节的统一接口，`main.py` 即可直接调用。

## 8. 论文列表

1. **Tensor-based Opposing yet Complementary Learning for Multi-view Multi-label Feature Selection** — [ACM MM 2025](https://dl.acm.org/doi/abs/10.1145/3746027.3755447)
2. **Uncertainty-Aware Global-View Reconstruction for Multi-View Multi-Label Feature Selection** — [AAAI 2025](https://github.com/hpinty/Multi-view-Multi-label-learning/tree/main/pre-pdf)
3. **Embedded feature fusion for multi-view multi-label feature selection** — [Pattern Recognition, 157 (2025) 110888](https://www.sciencedirect.com/science/article/pii/S0031320324006393)
4. **Double-Layer Hybrid-Label Identification Feature Selection for Multi-View Multi-Label Learning** — [AAAI 2024, 38(11): 12295-12303](https://ojs.aaai.org/index.php/AAAI/article/view/29120)
5. **Anchor-guided global view reconstruction for multi-view multi-label feature selection** — [Information Sciences, 679 (2024) 121124](https://www.sciencedirect.com/science/article/pii/S0020025524010387)
6. **Exploring view-specific label relationships for multi-view multi-label feature selection** — [Information Sciences, 681 (2024) 121215](https://www.sciencedirect.com/science/article/pii/S0020025524011290)

## 9. 引用与联系

请引用上述论文以支持本工作。

- 联系人：Pingting Hao（[862316425@qq.com](mailto:haopingting@nenu.edu.cn)，微信同号前缀）
- 单位：东北师范大学（Northeast Normal University）
