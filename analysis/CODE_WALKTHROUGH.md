# 六篇论文的代码逐块解读（`alg/*.py`）

> **阅读约定**
>
> * 严格按**文件内代码的先后顺序**（自上而下）讲；行号对应当前仓库版本。
> * **按要求跳过 import**；但各文件共同依赖 `alg/_util.py` 的三个小工具，先在第 0 节交代（否则后面每处都会卡）。
> * 每个文件先给一张**骨架表**（行区间 → 干什么），再逐块解读：**引一小段代码 → 逐句说明 → 对应论文哪个公式 → 已知的坑**。
> * 详细的公式解剖与偏差定级在 `PAPER_METHOD_ANALYSIS.md`（下称 **METHOD**）§3 各篇的"代码对照"与 §5.2；
>   本文只做"按代码顺序的导读"，遇到偏差处给一句结论 + 指针。
> * 顺序与 METHOD §3 一致：**TOCL → UGRFS → EF²FS → DHLI → GRAFS → I²VSLC**。
>   若想按"由易到难"读，建议 **I²VSLC(193 行) → UGRFS(173) → EF²FS(149) → DHLI(230) → GRAFS(196) → TOCL(288)**。

---

## 0. 公共零件：`alg/_util.py`

六篇的入口都被 `main.py` 用同一套参数调用，初始化与迭代上限全靠这几个函数：

| 函数 | 行 | 做什么 | 为什么需要它 |
|---|---|---|---|
| `DEFAULT_SEED = 100` | L10 | 默认随机种子 | 固定复现基线 |
| `make_rng(seed=None, default=100)` | L13–29 | 返回**独立的 `np.random.Generator`**；优先级：显式 `seed` > 环境变量 `MVML_SEED` > 默认 | 用独立 Generator 而不是全局 `np.random.seed`，避免污染其它库；TOCL 原来用全局 RNG 导致"同种子两次跑结果不同"（§5.2/REPRO §7.1） |
| `view_dim(x_view, i)` | L32–42 | 取第 $i$ 个视图的特征数 | `.mat` 里的 `features` 可能是 `(V,1)` 数组、也可能是扁平序列，这里统一成 `int` |
| `max_iter(default)` | L45–61 | 迭代上限，可用环境变量 `MVML_MAX_ITER` 覆盖 | 论文代码写死 500，全网格跑要几小时；这个开关让快速试跑成为可能（但会改变结果） |
| `as_int_indices(values)` | L64–71 | 把索引强制成 `int64` | 浮点索引会直接抛 `IndexError` |
| `dense(matrix)` | L74–80 | 稀疏矩阵转稠密，兼容 scipy/skfeature 的 API 变化 | `construct_W` 返回稀疏矩阵 |

---

## 1. TOCL — `alg/TOCL.py`（288 行）

### 1.1 骨架表

| 行区间 | 内容 |
|---|---|
| L15 | 常数 `eps` |
| **L17–68** | `prox_weight_tensor_nuclear_norm`：**加权 TNN 近端算子**（本文件的核心） |
| L70–75 | 入口 `view7()`：视图数、规模 |
| L77–79 | 三个**写死的系数**（`aaa`/`delta`/`rho`） |
| L81–92 | 按 `x_view` 把 `X` 切成视图块 `x[i]` |
| L94–112 | 初始化伪标签 `y[i]` 与选择矩阵 `d[i]` |
| L114–118 | 初始化 `W` |
| L120–134 | 初始化 `sum_yt`（各视图标签 OR）、`Y_n`、`P`、`Z`/`HH`、`kk`、`U`、`V` |
| L136–143 | 迭代计数与上限 |
| L144 | `while 1:` 主循环开始 |
| L146–158 | ① 更新 `y[i]`（伪标签） |
| L160–167 | ② `y[i]` 硬二值化 + 构造张量切片 |
| L169–179 | ③ 更新 `sum_yt` 与 **T**（调近端算子） |
| L181–192 | ④ 更新 `W` |
| L194–202 | ⑤ 更新 `P` |
| L204–209 | ⑥ 更新 `Y_n` |
| L211–218 | ⑦ 更新 `U`、`V` |
| L221–248 | 目标值与停机判据 |
| L264–285 | 排序与返回 |

### 1.2 逐块解读

**块 0｜L15**

```python
eps = 2.2204e-16
```
MATLAB 的 `eps`（float 机器精度）。六篇都用它做除零兜底。**注意它同时充当了论文式(24) 里的 $\epsilon$**（数值稳定项）。

---

**块 1｜L17–68 `prox_weight_tensor_nuclear_norm(Y, C)`**

这是全仓库最贵的一段。按顺序：

```python
L18:  n1, n2, n3 = Y.shape
L19:  X = np.zeros((n1, n2, n3), dtype=complex)      # 频域输出容器
L20:  Y = fft(Y, axis=2)                             # ← 沿第 3 维做 DFT
L21-22: wtnn = 0; trank = 0                          # 累积"加权核范数值"与管秩
```

* L20 是论文式(3) 的"DFT along the third dimension"。`scipy.fftpack.fft` 作用在 `axis=2` 上。
* `Y` 后面被就地覆盖成频域张量 —— 原张量不再需要，属于省内存的写法。

```python
L24-34:  # 注释写"第一个前切片"，但索引的是 [:, :, 1]
   U, S, Vh = svd(Y[:, :, 1], full_matrices=False)
   temp = (S - eps) ** 2 - 4 * (C - eps * S)        # 论文式(24) 的判别式 c2
   ind = np.where(temp > 0)[0]
   r = len(ind)
   S = np.maximum(S[ind] - eps + np.sqrt(temp[ind]), 0) / 2   # 式(24) 的 (c1+√c2)/2
   X[:, :, 1] = U[:, :r] @ np.diag(S) @ V[:, :r].T            # 式(25) 重构
   wtnn += np.sum(S * (C / (S + eps)))                        # 权重 w_i=C/(σ_i+ε)
   trank = max(trank, r)
```

* 逐句对应论文式(23) SVD → 式(24) 奇异值闭式解 → 式(25) 重构。
* **`j = 0` 这个切片从头到尾没被处理**（MATLAB 移植时索引从 1 开始）⇒ `X[:,:,0]` 恒为零矩阵，逆变换后丢掉直流分量（METHOD §5.2）。

```python
L36-49:  halfn3 = round(n3 / 2)
         for i in range(1, halfn3):
             ... 与上面完全相同的 SVD/收缩/重构 ...
             X[:, :, n3 - i] = np.conj(X[:, :, i])   # 共轭镜像
```

* 利用实信号频谱的共轭对称性：只算前一半切片，后一半取共轭，省一半 SVD —— 思路正确。
* **但顺序写错了**：循环体先写 `X[:,:,i]`，下一轮 `i+1` 又会算一遍自己的切片，而上一轮写的 `X[:,:,n3-i]` 会在**后续被覆盖**；同时 `wtnn` 的累加发生在覆盖**之前** ⇒ **累加到的切片集合与实际写进张量的切片集合不一致**（METHOD §5.2 M 行）。

```python
L51-63:  if n3 % 2 == 0:
             i = halfn3 + 1
             ... 对最后一个半切片做同样处理 ...
```
* 偶数长度时补一个"中间切片"。**当 `n3 = 2` 时 `i = 2` 越界 ⇒ IndexError**：这正是 2 视图数据集（emotions / yeast）直接崩的原因（METHOD §5.2 H 行；`insight/mechanisms.py::prox_weighted_tnn` 已修）。

```python
L65-68:  newX = np.fft.ifftn(X)
         wtnn /= n3
         return np.real(newX), wtnn, trank
```
* `wtnn /= n3` 对应论文式(22) 的 $1/n$ 归一化；但如前所述累加的切片数与 $n_3$ 不匹配 ⇒ 该值系统性偏大（它**不参与任何更新**，只进目标值记录，所以只影响日志）。
* 返回三元组：重构张量、加权核范数值、管秩上界（管秩被主循环忽略）。

> **一句话总结这个函数**：它是"论文式(22)–(25) 的实现 + 四个索引/MATLAB 移植缺陷"。修好的等价版本见 `insight/mechanisms.py::prox_weighted_tnn`（`python -m insight.run_demo M1` 会逐值对照）。

---

**块 2｜L70–79 入口与写死的系数**

```python
L70: def view7(X, x_view, Y, dataset, alpha, beta, gamma, lamb, seed=None):
L71:     rng = make_rng(seed)
L73-75:  time_start = time.time(); n_view = len(x_view); num, label_num = Y.shape
L76-79:  _, feature_num = X.shape
         aaa = 1; delta = 1; rho = 1
```

* 四个可调参数是 `alpha/beta/gamma/lamb`（**`lamb` 就是论文的 $\delta$**）。
* ⚠️ **`delta` 不是论文的 $\delta$**：这里的 `delta` 是**局部线性映射项的系数**（式(14) 里系数为 1 的那一项）；`aaa` 是张量项的系数；`rho` 是近端算子里的正则参数 $C$。三者都被写死为 1，论文从未说明（METHOD §5.2 L 行）。
* 论文的 4 个权衡系数在代码里对应 `alpha / beta / gamma / lamb`（`main.py::DEFAULT_PARAMS` 全为 1.0）。

---

**块 3｜L81–92 视图切分**

```python
for i in range(n_view):
    m.append(view_dim(x_view, i))
t1 = 0
for i in range(0, len(m)):
    x.append(X[:, :m[i]] if i == 0 else X[:, t1:(t1 + m[i])])
    t1 += m[i]
```

* `m[i]` 是第 $i$ 个视图的列数，`x[i]` 是从拼接矩阵 `X` 里切出来的 $X^{(i)}$。
* 六篇的切分代码几乎一模一样（只有 DHLI 用了 `m[i-1]` 的偏移写法）。

---

**块 4｜L94–112 初始化伪标签与选择矩阵**

```python
for i in range(n_view):
    y.append(rng.integers(2, size=(num, label_num)).astype(float))   # y⁽ⁱ⁾ 随机 0/1
    dd = np.zeros((m[i], feature_num))
    if i == 0:
        row, col = np.diag_indices(m[i]); dd[row, col] = 1
    else:
        col = as_int_indices(range(t2, t2 + m[i])); row = as_int_indices(range(m[i]))
        dd[row, col] = 1
    t2 += m[i]; d.append(dd)
```

* `y[i]` = 论文的 $y_t^{(i)}$（视图特有标签），随机二值初始化。
* `d[i]` = 论文的 $A^{(i)}\in\mathbb{R}^{d^{(i)}\times d}$：一个 **0/1 选择矩阵**，把全局权重 $W$ 的对应行块取出来（$U^{(i)}=A^{(i)}W$）。**它在整个训练中不再更新** —— 这是论文 Figure 2 的技巧，也是"跨视图共享 $W$"的机制所在。

---

**块 5｜L114–134 其余初始化**

```python
L118: W = rng.random((feature_num, label_num))     # ① 受种子控制的初始化
L121-126: sum_y = Σ_i y[i]; sum_yt = 1[sum_y > 0]  # ② 各视图标签的 OR（论文 Y_all）
L128-130: Y_n = rng.integers(2, ...)               # ③ 显式噪声标签
          P = rng.random((num, num))               # ④ 实例级耦合矩阵（稠密 n×n）
          Z = np.zeros((num, num, n_view)); HH = np.zeros_like(Z)
L132-134: kk = int(0.8 * label_num)                # ⑤ 隐标签维数
          U = rng.random((kk, label_num)); V = rng.random((feature_num, kk))
```

* ⓛ L118 的注释记录了一个**已修的种子泄漏**：原实现用全局 `np.random.rand`，导致 `--seed` 完全不起作用（METHOD §5.2 / REPRO §7.1）。
* ④ `P` 是**稠密 $n\times n$**：论文叫它"实例级重要性矩阵"，但它是样本方向的混合矩阵，不是对角权重（METHOD §3.1.2 的"两个容易看漏的细节"）。
* ⑤ `kk = floor(0.8·l)` 是论文未提的隐空间维数。
* `Z`/`HH`：`HH` 存 $T$ 的空域切片，`Z` 存近端算子的输出（低秩代理）。

---

**块 6｜L136–144 迭代准备**

```python
cver_lst = []; obj = []; obji = 1; iter = 0
MAX_ITER = max_iter(500)
I = np.ones((num, kk))          # 后面 sigmoid 分支的分母里当"全 1 矩阵"用
while 1:
```

* `I` 名字像单位阵，其实是**全 1 矩阵**，用在 $F=\mathbf 1/(\mathbf 1+e^{-XV})$ 的写法里（论文的 $\frac{I}{I+e^{-XV}}$ 照抄了矩阵记号）。

---

**块 7｜L146–158 更新 `y[i]`（论文式(21)）**

```python
for i in range(n_view):
    y_o = sum_yt - y[i]                     # 其他视图标签的 OR
    y_o[y_o == 0] = 0; y_o[y_o > 0] = 1
    F = Y - Y_n - y_o                       # 残差
    F[F == 0] = 0; F[F > 0] = 1; F[F < 0] = 1   # → 0/1 指示器
    y[i] = np.multiply(y[i], np.true_divide(
        delta * np.dot(P, np.dot(x[i], np.dot(d[i], W))) + aaa * np.dot(Z[:, :, i], y[i]) + gamma * F,
        delta * y[i] + gamma * y[i] + aaa * np.dot(np.dot(y[i], y[i].T), y[i]) + eps))
```

* 分子三块分别来自：**局部线性项**（$PX^{(i)}A^{(i)}W$）、**张量项**（$Z_{:,:,i}y^{(i)}$）、**离散标签相关项**（$\gamma F$）。
* 分母的 `aaa * (y[i] y[i]ᵀ) y[i]` 是张量罚项对 $y_i$ 的导数（出现两次乘积）。
* 这是**六篇通用的乘性更新**：分子=负梯度部分、分母=正梯度部分（通用推导见 METHOD §1.5.3）。

---

**块 8｜L160–167 硬二值化 + 构造张量切片**

```python
for i in range(n_view):
    y[i] = MinMax(y[i]); y[i][y[i] <= 0.5] = 0; y[i][y[i] > 0.5] = 1   # ← 论文没有这一步
    HH[:, :, i] = np.dot(y[i], y[i].T)                                # T 的第 i 个前切片
```

* 硬二值化：与 DHLI / I²VSLC 同一处问题 —— 乘性更新之间插入硬投影，破坏"目标单调"的前提（METHOD §5.2）。
* `HH[:,:,i] = y yᵀ` 就是论文式(10) 里的 $T_{:,:,i}$（标签 Gram 矩阵）。

---

**块 9｜L169–179 更新 `sum_yt` 与张量 `T`**

```python
sum_y = Σ_i y[i]; sum_yt = 1[sum_y > 0]              # 硬二值化后重算 OR
HH2 = HH.transpose((0, 2, 1))                        # (n,n,V) → (n,V,n)
Z2, wtnn, trank = prox_weight_tensor_nuclear_norm(HH2, rho)   # rho = C = 1
Z = Z2.transpose((0, 2, 1))                          # 转回来
```

* 这两次转置 + 近端算子内部对 `axis=2` 做 FFT ⇒ **FFT 实际沿样本方向**，与"刻画跨视图相关性"的意图不符（METHOD §5.2 TOCL 的 H 行；机制与修法见 `insight/mechanisms.py` 的 `fft_axis` 参数）。
* 这一步是整个算法的时间瓶颈：每轮一次 FFT + 约 $n/2$ 次矩阵 SVD。

---

**块 10｜L181–192 更新 `W`（论文式(16)）**

```python
NL = 1 / (I + np.exp(-np.dot(X, V)))                 # F = sigmoid(XV)，n×k
d_temp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps) # 行 2-范数
D2 = np.diag((0.5 / d_temp).flat)                    # ℓ2,1 的重加权对角阵
tem1 = Σ_i d[i].T x[i].T P.T y[i]
tem2 = Σ_i (d[i].T x[i].T P.T P x[i] d[i]) W
W = W ∘ (delta*tem1 + alpha*X.T P.T NL U) / (delta*tem2 + lamb*D2 W + alpha*X.T P.T P X W + eps)
```

* 分子=局部项 + 全局非线性项；分母=对应抑制项 + **$\ell_{2,1}$ 稀疏项**（`lamb` 即论文 $\delta$）。
* `D2` 就是 METHOD §1.5.2 讲的"辅助对角阵"：`0.5/‖W_row‖`。

---

**块 11｜L194–202 更新 `P`（论文式(17)）**

```python
tem1 = Σ_i y[i] (d[i]W).T x[i].T
tem2 = Σ_i P x[i] (d[i]W)(d[i]W).T x[i].T
A = np.dot(P, X)
P = P ∘ (delta*tem1 + alpha*NL U W.T X.T) / (delta*tem2 + beta*P + alpha*A W W.T X.T + eps)
```
* 分子两项正是论文 §3.1.3 说的"对立（局部）+ 互补（全局）"在更新式里的代数痕迹；分母末项 `beta*P` 来自 $\beta\|P\|_F^2$。

---

**块 12｜L204–218 更新 `Y_n`、`U`、`V`（论文式(18)(19)(20)）**

```python
L205-209: tem3 = 1[Y - sum_yt ≠ 0];  Y_n = Y_n ∘ tem3 / (2*Y_n + eps)
L212-213: NL = sigmoid(XV);  U = U ∘ (NL.T P X W) / (NL.T NL U + 2U + eps)
L216-218: Q = e^{-XV} / (1 + e^{-XV})²          # sigmoid 的导数 σ'
          V = V ∘ (X.T (Q ∘ (P X W U.T))) / (X.T (Q ∘ (NL U U.T)) + 2V + eps)
```
* `U` 与 `V` 的分母里有 `2U`/`2V`，来自正则 $\|U\|_F^2+\|V\|_F^2$ 的导数。
* `Q` 是链式法则的产物：$\partial\,\mathrm{sigmoid}/\partial(XV)$。
* ⚠️ `Y_n` 的**更新式**用 $Y-Y_{all}$，而**目标值**（下一块）用 $Y_{all}+Y_n-Y$，两者判据不同步（METHOD §3.1.5 的差异表）。

---

**块 13｜L221–248 目标值与停机判据**

```python
temp1 = Σ_i ‖P x[i] d[i] W − y[i]‖_F²          # ① 局部线性
temp2 = Σ_i ‖Z[:,:,i] − HH[:,:,i]‖_F²          # ③ 张量拟合
temp3 = 0.5*temp2 + wtnn                        #    加权 TNN 值
temp4 = ‖P X W − NL U‖_F²                       # ② 全局非线性
temp5 = ‖U‖_F² + ‖V‖_F²
tem7  = 1[sum_yt + Y_n − Y ≠ 0]; temp7 = ‖tem7‖²; temp8 = ‖Y_n‖_F²   # ④ 离散标签相关
objectives = delta*temp1 + aaa*temp3 + alpha*(temp4+temp5) + 2*beta*temp6 + gamma*(temp7+temp8) + 2*lamb*temp9
```
之后：

```python
if iter == 0: obji = objectives / 2          # ← 让首轮相对变化率 ≈ 1（论文 Fig.5 的"初值≈1"）
obj.append(objectives)
cver = abs((objectives - obji) / float(obji)); obji = objectives
iter += 1
if (iter > 2 and (cver < 1e-6 or iter == MAX_ITER)): break
```
* **停机阈值 $10^{-6}$**，比其它五篇的 $10^{-3}$ 严格两个数量级 ⇒ TOCL 迭代轮数最多（`3sources` 6 轮、`MIRFlickr` 跑满上限）。
* `objectives/2` 这一手解释了论文 Fig.5 里"初值约为 1"（METHOD §5.2 L 行）。
* 目标值里 `2*beta`、`2*lamb` 多出因子 2，与论文式(14) 不符（只影响停机判据）。

---

**块 14｜L264–285 排序与返回**

```python
w_2 = LA.norm(W, ord=2, axis=1)      # 逐行 2-范数 = 特征重要性
f_idx = np.argsort(-w_2)             # 降序
record = {method, dataset, running_time, selected_num, param, obj_value, idx}
return record, iter
```
* 与六篇完全一致：**排序量 = $W$ 的行 2-范数，降序**。
* 返回的 `record` 由 `main.py` 消费（`idx` 是特征索引顺序）。

---

## 2. UGRFS — `alg/UGRFS.py`（173 行）

### 2.1 骨架表

| 行区间 | 内容 |
|---|---|
| L10 | `eps` |
| L12–28 | `kernelmatrix()`：高斯核 |
| L30–58 | 入口与初始化（`c[i]` 样本置信度、`w[i]`、`x[i]`） |
| L60–71 | **标签图**拉普拉斯 `Ly` |
| L73–84 | **视图权重** `nu` 与加权拼接 `new_X` |
| L86–89 | 标签核映射 `Yx = [H, 1]` |
| L91–100 | `wy[i]` 初始化与迭代上限 |
| L102–120 | 主循环：`w[i]` → `c[i]` → `wy[i]` |
| L123–138 | 目标值 |
| L140–150 | 停机判据 |
| L152–172 | 排序、记录、返回 |

### 2.2 逐块解读

**块 0｜L12–28 `kernelmatrix(par, trainX, testX)`**

```python
D = 1ᵀn1sq + 1 n2sqᵀ − 2·testX·trainXᵀ      # 两两平方欧氏距离
H = np.exp(-D / (2 * np.square(par)))
```
* 标准高斯核；`par` 是**带宽**。用"全 1 向量"广播算距离矩阵（MATLAB 风格，$n^2$ 内存）。

---

**块 1｜L30–58 入口与初始化**

```python
L42-45: for i in range(n_view):
            m.append(view_dim(x_view, i))
            c.append(np.diag(rng.random((num, 1)).flat))     # C⁽ⁱ⁾：样本置信度（对角）
L57-58: for i in range(n_view):
            w.append(rng.random((x[i].shape[1], label_num))) # W⁽ⁱ⁾
```
* `c[i]` 就是论文的 $C^{(i)}$（样本级不确定度），初始化在 $(0,1)$；`np.diag(随机向量)` 把它变成 $n\times n$ 对角阵。
* `w[i]` 是**每个视图各一份**权重（论文的 $W^{(i)}$，$d^{(i)}\times l$）—— 与 TOCL"共享一个全局 W"形成对照。

---

**块 2｜L60–71 标签图**

```python
options = {'metric':'euclidean','neighbor_mode':'knn','k':20,'weight_mode':'heat_kernel','t':1.0}
Sy = dense(construct_W(Y, **options))     # ← 图建在**标签**上
Ay = np.diag(np.sum(Sy, 0)); Ly = Ay - Sy
```
* 这是 UGRFS 与其它篇的第一个结构性差异：**图建在标签空间**（$Y$ 为节点特征），而 I²VSLC/EF²FS 建在视图特征上（METHOD §1.5.1 的对照表）。
* `t=1.0` 写死 —— §1.4 的热核下溢陷阱就来自这一类代码。

---

**块 3｜L73–84 视图权重与加权拼接**

```python
for i in range(n_view):
    nu_temp[i] = (1 / np.trace(x[i].T @ Ly @ x[i]))      # 图能量倒数
    nu_all += nu_temp[i]
nu = nu_temp / nu_all                                     # 归一化
new_X = np.concatenate([x[i] * nu[i] for i in range(n_view)], axis=1)   # X^f
```
* 对应论文的 $v_i\propto 1/\mathrm{Tr}(X^{(i)\top}L_yX^{(i)})$。
* ⚠️ 论文写的是 $v_i\propto1/\mathrm{Tr}(y^\top L_x^{(i)}y)$（标签 × **视图特征图**），代码是**视图特征 × 标签图** —— 角色互换（METHOD §5.2 H 行）；而且这段在 `while` **外面**，只算一次，与"随 $G$ 演化"的语义不符。
* $\mathrm{Tr}$ 值近 0 时除法会爆炸（`1/eps` 兜底），这是 §5.4 记录的权重塌缩机制。

---

**块 4｜L86–89 标签核映射**

```python
par = 1 / np.mean(pdist(Y))        # ← 带宽 = 1/平均距离
H = kernelmatrix(par, Y, Y)
Yx = np.concatenate([H, np.ones((num, 1))], axis=1)     # [ρ(Y), 1] ∈ n×(n+1)
```
* 对应论文式(6) 的 $\rho(Y)$ 与式(7) 的 $D=Y_xW_y$。`Yx` 比论文少一个偏置列的处理（这里把偏置**拼成一列常数 1**，等价于线性回归的截距）。
* ⚠️ 论文的 $\rho(Y)$ 分子是 $J_1Y^\top(\sum_j y_{ij}^2)-YY^\top$，代码换成了标准高斯核，且带宽方向相反（$\sigma=\text{avg}$ vs 代码 $1/\text{avg}$）—— METHOD §5.2 H 行。

---

**块 5｜L102–120 主循环（按 `w[i]` → `c[i]` → `wy[i]` 的顺序）**

```python
L106-108: d_temp = sqrt(Σ_j w[i]²_j + eps);  D = diag(0.5 / d_temp)
L110:     p1 = np.dot(c[i], x[i])                       # A⁽ⁱ⁾ = diag(C⁽ⁱ⁾)X⁽ⁱ⁾
L111-112: w[i] = w[i] ∘ (p1.T Y) / (p1.T p1 w[i] + lamb * D w[i] + eps)     # 式(14)
```
* 分子 `p1.T Y`：置信度加权后的特征与标签的相关；分母是重构项 + $\ell_{2,1}$ 重加权项。
* ⚠️ `0.5/‖·‖` 的实现使有效稀疏权重只有论文 $2\delta$ 的一半（METHOD §5.2 M 行）。

```python
L114: c[i] = c[i] ∘ (beta*(Yx wy[i]) x[i].T + (Y w[i].T) x[i].T)
                    / (beta*(c[i] x[i]) x[i].T + ((p1 w[i]) w[i].T) x[i].T + eps)
L115-117: c[i] = np.diagonal(c[i]).reshape(num,1); c[i] = np.diag(c[i].flat)
```
* 对应论文式(15)。**L115–117 是"强制对角"**：乘性更新出来的 $n\times n$ 矩阵只取对角线，再变回对角阵 ⇒ 保证 $C^{(i)}$ 始终是"逐样本权重"。
* ⚠️ 分母里的 `p1` 是**旧的** `c[i]@x[i]`，而分子用的是刚更新的 `w[i]`（Jacobi 式混合，论文未声明）。

```python
L119-120: p3 = x[i] * nu[i]
          wy[i] = wy[i] ∘ (gamma*Yx.T p3 + alpha*Yx.T Sy Yx wy[i] + beta*Yx.T c[i] x[i])
                          / (alpha*Yx.T Ay Yx wy[i] + (gamma+beta)*Yx.T Yx wy[i] + eps)
```
* 对应论文式(16)：分子三项 = 与 $X^f$ 一致（γ）、图上平滑（α，由图 $\{Sy,Ay\}$ 给出）、拆回各视图（β）。
* `wy[i]` 是每个视图的 $W_y^{(i)}$；`p2 = Yx @ Wy`（下一块）才是全局视图 $D$。

---

**块 6｜L123–138 目标值**

```python
Wy = concat(wy, axis=1)
temp1 = ‖Yx Wy − new_X‖_F²                 # γ：D 与加权拼接一致
p2 = Yx Wy;  temp2 = Tr(p2.T Ly p2)        # α：D 的标签图平滑
temp3 = Σ_i ‖Yx wy[i] − c[i] x[i]‖²        # β：拆回各视图
temp4 = Σ_i ‖c[i] x[i] w[i] − Y‖²          # 主项：视图→标签
W = concat(w, axis=0);  temp5 = Σ_p ‖W_p‖₂ # δ：ℓ2,1
objectives = temp4 + alpha*temp2 + beta*temp3 + gamma*temp1 + lamb*temp5
```
* 五项与论文式(12) 一一对应（`lamb` 即 $\delta$）。

**块 7｜L140–150 停机**

```python
if iter == 0: obji = objectives / 2
cver = abs((objectives - obji) / float(obji))
if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)): break
```
* `objectives/2` ⇒ 首轮 `cver = 1`，正对应论文 Fig.5 的"初值约为 1"（论文正文完全没解释）。
* 停机阈值 $10^{-3}$，上限 500。

**块 8｜L152–172 排序与返回**

```python
W = concat(w, axis=0)                 # 纵向拼接成 d×l
w_2 = LA.norm(W, ord=2, axis=1);  f_idx = np.argsort(-w_2)
```
* 排序量仍是全局权重（拼接后）的行 2-范数。

---

## 3. EF²FS — `alg/EF2FS.py`（149 行）

### 3.1 骨架表

| 行区间 | 内容 |
|---|---|
| L21 | `eps` |
| L24–36 | 入口与初始化（`w[i]`、`v[i]` 视图块） |
| L43–50 | 每个视图的图拉普拉斯 `Lx_lst` |
| L52–54 | 隐标签 `V`、`B`、`A` 初始化 |
| **L61–77** | **每轮第一步：视图权重 `nu` 与加权拼接 `new_X`** |
| L79–102 | `B` → `V` → `A` → `w[i]` 的乘性更新 |
| L104–120 | 目标值 |
| L121–130 | 停机 |
| L132–149 | 排序、记录、返回 |

> ⚠️ 先记住一个命名陷阱：**代码的 `V` 是论文的隐标签 $G$**，`B` 是论文的 $B$，`A` 是排序依据（论文的 $A$），`w[i]` 是论文的 $w^{(i)}$。

### 3.2 逐块解读

**块 0｜L24–41 入口、初始化、切分**

```python
L24: def EF2FS(X, x_view, Y, dataset, alpha, beta, gamma, lamb, V_dim, seed=None):
L34-36: for i in range(n_view):
            m.append(view_dim(x_view, i)); w.append(rng.random((m[i], V_dim)))
L38-41: 切分视图块 v[i]
```
* 多了一个**结构参数 `V_dim`**（隐空间维数 $k$，`main.py` 默认 30）—— 论文全文没给这个值（METHOD §5.2 M 行）。

**块 1｜L43–50 图拉普拉斯（在循环外一次性建好）**

```python
options = {'k': 5, 'weight_mode':'heat_kernel', 't': 1.0, ...}
for i in range(n_view):
    Sx = dense(construct_W(v[i], **options)); Ax = diag(Σ Sx); Lx_lst.append(Ax - Sx)
```
* 建在**视图特征**上（与 UGRFS 相反）。
* 文件头注释说明：原实现把视图权重那段写在 `Lx_lst` 定义**之前**，第一次迭代直接 `NameError`；这里是修好的顺序。

**块 2｜L52–54 初始化**

```python
V = rng.random((num, V_dim))       # ← 论文的 G（隐标签/公共嵌入）
B = rng.random((label_num, V_dim))
A = rng.random((dim, V_dim))       # ← 排序依据
```

**块 3｜L61–77 每轮先算视图权重**

```python
for i in range(n_view):
    energy = float(np.trace(V.T @ Lx_lst[i] @ V))       # E_i = Tr(Gᵀ L⁽ⁱ⁾ G)
    nu_temp[i] = 1.0 / energy if energy > eps else 0.0  # 除零保护（原实现没有）
nu = nu_temp / nu_all  (全 0 时退化为均匀权重)
new_X = np.concatenate([v[i] * nu[i] for i in range(n_view)], axis=1)   # X^f
```
* 这是论文式(12) 的 $c_i\propto1/E_i$。**注意它用的是上一轮的 `V`（$G$）**——式(12) 定义在当前 $G$ 上，代码滞后一轮（METHOD §5.2 M 行）。
* `X^f` 在这里叫 `new_X`，只在 `V` 的更新与目标值里用。

**块 4｜L79–102 四个变量的乘性更新（顺序：B → G → A → w）**

```python
L80: B = B ∘ (Y.T V) / (B V.T V + eps)                       # 式(23)
L82-84: xw = Σ_i v[i] w[i]
L86-88: V = V ∘ (alpha*Y B + beta*xw + new_X A) / (alpha*V B.T B + (n_view*beta + 1)*V + eps)   # 式(21)
```
* `V`（即 $G$）的分子三项正好对应目标里的 ①②③（标签回归、各视图拟合、融合重构）—— 是"隐标签被三面牵引"的直接体现。

```python
L90-91: Atmp = sqrt(Σ A²_row + eps);  D = diag(0.5 / Atmp)
L92-94: A = A ∘ (X.T V + gamma * concat(w)) / (X.T X A + gamma*A + lamb*D A + eps)   # 式(25)
```
* ⚠️ **分子分母都用未加权的 `X`**，而 $G$ 的更新与目标值都用加权的 `new_X` ⇒ 这一步的下降方向与记录的目标函数不一致（METHOD §5.2 H 行）。
* `lamb*D A` 是 $\ell_{2,1}$ 项（`lamb` = 论文 $\delta$）。

```python
L96-102: for i in range(n_view):
             new_a = A[t1:t1+m[i], :]                        # A 的第 i 个行块 = a⁽ⁱ⁾
             w[i] = w[i] ∘ (beta*v[i].T V + gamma*new_a) / (beta*v[i].T v[i] w[i] + gamma*w[i] + eps)
```
* 对应论文式(27)：分子两项 = 拟合 $G$（β）+ 与全局块一致（γ）。

**块 5｜L104–120 目标值**

```python
temp1 = ‖new_X A − V‖²        # ① 融合→嵌入
temp2 = ‖Y − V B.T‖²          # ② 嵌入→标签
temp3 = Σ_i ‖v[i] w[i] − V‖²  # ③ 各视图→嵌入
temp4 = Σ_i ‖A 的第 i 块 − w[i]‖²   # ④ 全局块↔局部权重
temp5 = Σ_p ‖A_p‖₂            # ⑤ ℓ2,1
objectives = temp1 + alpha*temp2 + beta*temp3 + gamma*temp4 + lamb*temp5
```
* 与论文式(12) 一一对应。

**块 6｜L121–149 停机、排序、返回**

```python
if iter == 0: obji = objectives / 2
cver = abs((objectives - obji) / float(obji))
if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)): break
w_2 = LA.norm(A, ord=2, axis=1);  f_idx = np.argsort(-w_2)     # 排序用 A
```
* `MAX_ITER = max_iter(100)` —— 六篇里唯一把上限设成 100 的（其它是 500），这也是它跑得快的原因之一。
* 停机用了 `abs()`，而论文 Fig.4 的纵轴是**带符号**的比值（METHOD §5.2 L 行）。

---

## 4. DHLI — `alg/DHLI.py`（230 行）

### 4.1 骨架表

| 行区间 | 内容 |
|---|---|
| L10 | `eps` |
| L13–27 | 入口与初始化（`w[i]`、`u[i]`、`y[i]`） |
| L29–35 | `sum_yt`（各视图标签 OR） |
| L37–43 | 视图切分 |
| L45–57 | `Y_c`、`Y_n` 随机初始化；`Y_spe` 首次计算 |
| L62–95 | 视图循环：`w[i]`、`u[i]`、`y[i]` |
| L98–103 | `y[i]` 硬二值化 |
| L105–111 | 重算 `sum_yt` |
| L114–124 | `Y_c` 更新 + 硬二值化 |
| L127–140 | `Y_n` 更新 + 硬二值化 |
| L143–184 | 目标值 |
| L185–196 | 停机 |
| L198–212 | 排序（`MinMax(W)+MinMax(U)`） |
| L214–230 | 记录、返回 |

### 4.2 逐块解读

**块 0｜L13–27 初始化**

```python
for i in range(n_view):
    w.append(rng.random((m[i], label_num)))    # W⁽ⁱ⁾ → 拟合 Y_c
    u.append(rng.random((m[i], label_num)))    # U⁽ⁱ⁾ → 拟合 y_s⁽ⁱ⁾
    y.append(rng.integers(2, size=(num, label_num)))   # 视图特有标签
```
* **双权重**：`w[i]` 学公共标签 $Y_c$，`u[i]` 学视图特有标签 $y_s^{(i)}$ —— 这是 DHLI 与其它篇最大的结构差别。
* `y[i]` 是 `int` 数组（不是 float，后面参与乘性更新时会被隐式转换）。

**块 1｜L29–43 `sum_yt` 与切分**

```python
sum_y = Σ_i y[i];  sum_yt = 1[sum_y > 0]      # Y_all = 各视图标签的 OR
...
x.append(X[:, :m[i]] if i == 0 else X[:, m[i-1]:(m[i-1] + m[i])])
```
* 切分用了 `m[i-1]` 的偏移写法（其它文件用累计变量 `t1`），等价但容易看错。
* `sum_yt` 就是论文里的 $Y_{all}=\bigotimes_i y_s^{(i)}$（OR 聚合）。

**块 2｜L45–57 三个标签矩阵的初始化**

```python
Y_c = rng.integers(2, size=(num, label_num)).astype(float)   # 公共标签
Y_n = rng.integers(2, size=(num, label_num)).astype(float)   # 噪声
Y_spe = Y - Y_c - Y_n
Y_spe[Y_spe <= 0] = 0; Y_spe[Y_spe > 0] = 1                  # ← 负差归 0
```
* 对应论文 Definition 1 的三类切分。**注意这里的二值化规则（负差归 0）与循环内那处（负差归 1）不一致**（METHOD §5.2 H 行）。

**块 3｜L62–95 视图循环**

```python
L63-69: d_temp = sqrt(Σ_j w²+eps); D = diag(0.5/d_temp)     # ℓ2,1 重加权（W）
        e_temp = sqrt(Σ_j u²+eps); E = diag(0.5/e_temp)     # ℓ2,1 重加权（U）
L71-72: w[i] = w[i] ∘ (x[i].T Y_c) / (x[i].T x[i] w[i] + gamma*D w[i] + eps)   # 式(16)
L73-74: u[i] = u[i] ∘ (x[i].T y[i]) / (x[i].T x[i] u[i] + gamma*E u[i] + eps)  # 式(17)
```
* 两个更新形状完全一样，只是目标标签不同（$Y_c$ vs $y_s^{(i)}$）。`gamma` 在这里是稀疏权重。

```python
L76-78: y_o = 1[sum_yt − y[i] ≠ 0]        # 其他视图的 OR（留一）
L84-87: B = 1
        for j in range(n_view):
            if i != j: B = B * y[j]        # 其他视图的 AND
```
* **留一量**：`y_o` 对应论文的 $\bigotimes_{j\ne i}y_s^{(j)}$，`B` 对应 $\bigcirc_{j\ne i}y_s^{(j)}$。
* L80–83 的注释记录了一个**已修的上游 bug**：原实现 `B = 0; B = B * y[i]` 恒为 0（且含自身），互斥项静默失效；现在从乘积单位元 1 开始、且跳过自身。

```python
L89-92: tem1 = 1[Y_spe − y_o ≠ 0]
L93:    y[i] = y[i] ∘ (x[i] u[i] + lamb*tem1) / (y[i] + lamb*y[i] + lamb*y[i]*B*B + eps)
```
* 分母的 `lamb*y[i]*B*B` 就是**视图互斥**项 $\delta\|\bigcirc_i y_s^{(i)}\|^2$ 对 $y_i$ 的导数（乘积出现两次）；分子 `lamb*tem1` 来自并集一致项。
* 这里 `lamb` 是论文的 **$\delta$**（不是稀疏项，稀疏在 `gamma`）—— 这正是 §5.3 参数表里 DHLI 那一行的例外。

```python
L95: t1 += np.dot(x[i], w[i])     # A = Σ_i X⁽ⁱ⁾W⁽ⁱ⁾，在视图循环内累加
```
* ⚠️ 在循环内累加 ⇒ 一次迭代里**混用了已更新与未更新的 `w[i]`**（METHOD §5.2 M 行）。`t1` 后面给 $Y_c$ 的更新用。

**块 4｜L98–111 硬二值化与 `sum_yt` 重算**

```python
for i in range(n_view):
    y[i] = MinMax(y[i]); y[i][y[i] <= 0.5] = 0; y[i][y[i] > 0.5] = 1
sum_y = Σ y[i]; sum_yt = 1[sum_y > 0]
```
* 论文完全没有这一步。后果：变量被强制成 $\{0,1\}$、**每列必然至少有一个 1**（MinMax 的性质）、$0$ 成为乘性更新的吸收态、破坏"目标单调"的前提（METHOD §5.2 H 行）。

**块 5｜L114–124 `Y_c` 更新**

```python
tem2 = 1[Y − Y_n − sum_yt ≠ 0]
Y_c = Y_c ∘ (t1 + alpha*Y + lamb*tem2) / (n_view*Y_c + alpha*Y_c + lamb*Y_c + eps)   # 式(18)
Y_c = MinMax(Y_c); Y_c[Y_c <= 0.5] = 0; Y_c[Y_c > 0.5] = 1
```
* 分子三项=各视图预测之和 `t1` + 观测标签先验 `alpha*Y` + 修正项 `lamb*tem2`；分母的 `n_view*Y_c` 是 $V$ 份证据的归一化。
* `lamb` 在这里又是"并集/离散项"的权重。

**块 6｜L127–140 `Y_n` 更新**

```python
tem3 = 1[Y − Y_c − sum_yt ≠ 0]
Q = 1 / (2 * abs(Y_n) + eps)                       # 论文的自适应 ℓ1 重加权
Y_n = Y_n ∘ (lamb*tem3) / (beta*Q*Y_n + lamb*Y_n + eps)      # 式(19)
Y_n = MinMax(Y_n); Y_n[Y_n <= 0.5] = 0; Y_n[Y_n > 0.5] = 1
```
* ⚠️ `Y_n` 被硬二值化后 $Q\equiv1/2$，于是 `beta*Q*Y_n` 退化成常数 $\beta/2$ —— 论文声称的"自适应 $\ell_1$ 重加权"**名存实亡**（METHOD §5.2 M 行）。

**块 7｜L143–184 目标值**

```python
temp1 = Σ_i ‖x[i] w[i] − Y_c‖²        # 主项①：公共标签
temp2 = Σ_i ‖x[i] u[i] − y[i]‖²       # 主项②：视图特有标签
temp3 = alpha * ‖Y − Y_c‖²            # 公共占多数
temp4 = beta * 2 * Σ|Y_n|             # 噪声稀疏（注意多了一个 2）
temp5/6 = gamma * 2 * Tr(WᵀD_all W) / Tr(UᵀE_all U)     # ℓ2,1（同样带 2）
temp7 = lamb * ‖∏_i y[i]‖²            # 视图互斥
temp8 = lamb * ‖1[sum_yt − Y_spe ≠ 0]‖²                  # 并集一致
objectives = temp1+temp2+temp3+temp4+temp5+temp6+temp7+temp8
```
* 逐项对应论文式(12)。**目标值里 β 项与 γ 项都多一个因子 2**，只影响停机判据（METHOD §5.2 M 行）。
* 注意 L173–176 又算了一次 `Y_spe`，这次**负差归 1**（与 L55–57 不一致）。

**块 8｜L185–196 停机**

```python
if iter == 0: obji = objectives / 2
cver = abs((objectives - obji) / float(obji))
if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)): break
```
* 与其它篇相同的"首轮 cver≈1"技巧。

**块 9｜L198–212 排序：两组权重相加**

```python
W = concat(w, axis=0);  U = concat(u, axis=0)
tW = MinMax(W);  tU = MinMax(U)
A = tW + tU
w_2 = LA.norm(A, ord=2, axis=1);  f_idx = np.argsort(-w_2)
```
* 论文 Algorithm 1 只写 $\|(W+U)_{(j)}\|_2$；代码**先各自 MinMax 再相加**，多了一步归一化（METHOD §5.2 M 行）。

---

## 5. GRAFS — `alg/GRAFS.py`（196 行）

### 5.1 骨架表

| 行区间 | 内容 |
|---|---|
| L15 | `eps` |
| L17–19 | `normalization()`：MinMax |
| L22–56 | 入口与初始化（`k1=10`、`B`、`W`、`A1/A2`、`W1`、`X1`、`R1/R2`） |
| L58–77 | 视图循环：`s[i]`（=C⁽ⁱ⁾）、`d[i]`（=D⁽ⁱ⁾ 选择矩阵）、`p[i]`（=P⁽ⁱ⁾） |
| L79–96 | **标签图** `Ly` 与视图权重 `nu` |
| L104–122 | 主循环：`s` → `p` → `W` → `R1/R2/A1/A2/W1` |
| L124–138 | `B` 与 `X1`（全局视图）的更新 |
| L140–166 | 目标值、停机 |
| L169–195 | 计时、排序、记录、返回 |

### 5.2 逐块解读

**块 0｜L17–19 `normalization()`**

```python
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
```
* 全局 MinMax（不是逐列）；`_range = 0` 时除零（无保护）。

**块 1｜L22–56 入口与初始化**

```python
L22: def view6(X, x_view, Y, dataset, alpha, beta, gamma, lamb, kk, seed=None):
L46: k1 = 10                                    # ← 潜锚点维数，硬编码
L47: B = rng.random((num, kk))                  # 候选视图 B ∈ n×k（kk 默认 20）
L48: W = rng.random((feature_num, label_num))   # 排序依据 W ∈ d×l
L49-50: A1 = rng.random((k1, k1)); A2 = rng.random((k1, k1))   # A^c, A^f
L51: W1 = rng.random((num, k1))                 # W^c，两条路径**共享**
L52-53: X1 = normalization(np.dot(Y, W.T))      # 全局视图 X^f 用 YWᵀ 暖启动（论文未提）
L55-56: R1 = A1 @ W1.T @ B;  R2 = A2 @ W1.T @ X1
```
* 变量对照：**`X1`=全局视图 $X^f$（$n\times d$）、`B`=候选视图（$n\times k$）、`W1`=$W^c$、`A1/A2`=$A^c/A^f$、`R1/R2`=$R^c/R^f$、`s[i]`=$C^{(i)}$、`d[i]`=$D^{(i)}$、`p[i]`=$P^{(i)}$、`W`=排序矩阵**。
* ⚠️ `k1` 硬编码 10、`kk` 由调用者传（默认 20）—— **论文这两个结构参数都没报**（METHOD §5.2 H 行）。
* `X1` 的维度是 $n\times d$（与拼接特征同维），不是低维潜空间；因为排序矩阵 $W$ 必须是 $d\times l$。

**块 2｜L58–77 视图循环**

```python
for i in range(n_view):
    s1 = rng.random((m[i], 1)); s1 = s1 / np.sum(s1)
    s.append(np.diag(s1.flat))        # C⁽ⁱ⁾：被强制成对角阵
    dd[row, col] = 1                  # D⁽ⁱ⁾：d×d⁽ⁱ⁾ 选择矩阵（列块取出）
    p.append(np.dot(B.T, x[i]))       # P⁽ⁱ⁾：k×d⁽ⁱ⁾
```
* ⚠️ **$C^{(i)}$ 被强制对角**，参数量从 $O(d^{(i)2})$ 降到 $O(d^{(i)})$，无法建模特征间交互（论文是满矩阵）—— METHOD §5.2 H 行。
* `D⁽ⁱ⁾` 是 0/1 选择矩阵，等价于"把 $X^f$ 的第 $i$ 个列块切出来"，即 $\|X^fD^{(i)}-X^{(i)}C^{(i)}\|$ 里的切分算子。

**块 3｜L79–96 标签图与视图权重**

```python
Sy = dense(construct_W(Y_ori, **options))     # ← 建在**标签**上（论文要求视图特征图）
Ly = Ay - Sy
for i in range(n_view):
    nu_temp[i] = 1 / np.trace(x[i].T @ Ly @ x[i])
nu = nu_temp / nu_all
```
* ⚠️ 论文式(2)(3)(5) 的**视图**亲和图 $S^{(v)}/L_x^{(v)}$ 完全没实现，换成了标签图的转置用法（METHOD §5.2 H 行）。
* 仍然构造 $n\times n$ 稠密标签图（`OBJECT` 的 $n=6047$ 时 `Sy/Ay/Ly` 合计约 0.9 GB），抵消了"用锚点图替代全图"的动机（METHOD §5.2 M 行）。

**块 4｜L104–122 主循环前半：`s` → `p` → `W` → 锚点路径**

```python
L106-109: s[i] = s[i] ∘ (x[i].T X1 d[i]) / (x[i].T x[i] s[i] + eps);  取对角再 diag
L110:     p[i] = p[i] ∘ (B.T x[i]) / (B.T B p[i] + eps)
L112-115: E = diag(0.5/‖W_row‖);  W = W ∘ (X1.T Y) / (X1.T X1 W + lamb*E W + eps)
```
* `W` 的更新对应论文式(25)：分子是全局视图与标签的相关。

```python
L118: R1 = R1 ∘ (A1.T W1.T B) / (A1.T W1.T W1 A1 R1 + eps)          # R^c
L119: R2 = R2 ∘ (A2.T W1.T X1) / (A2.T W1.T W1 A2 R2 + eps)         # R^f
L120: A1 = A1 ∘ (W1.T B R1.T + A2) / (W1.T W1 A1 R1 R1.T + A1 + eps)   # ← 分子含 A2
L121: A2 = A2 ∘ (W1.T X1 R2.T + A1) / (W1.T W1 A2 R2 R2.T + A2 + eps)  # ← 分子含 A1
L122: W1 = W1 ∘ (B R1.T A1.T + X1 R2.T A2.T) / (W1 A1 R1 R1.T A1.T + W1 A2 R2 R2.T A2.T + eps)
```
* **这是 GRAFS 最精妙的一处落到代码的样子**：`A1`/`A2` 的更新里各出现对方 ⇒ 就是 $\|A^c-A^f\|_F^2$ 锚点软一致项的梯度；`W1`（$W^c$）的分子同时含两条路径 ⇒ **共享同一套簇坐标**（METHOD §3.5.3）。

**块 5｜L124–138 候选视图与全局视图的更新**

```python
q1 = Σ_i nu[i] x[i] p[i].T ;  q2 = Σ_i nu[i] (B p[i]) p[i].T
q3 = Σ_i nu[i] (x[i] s[i]) d[i].T ;  q4 = Σ_i nu[i] (X1 d[i]) d[i].T
L135: B  = B  ∘ (alpha*q1 + beta*W1 A1 R1) / (alpha*q2 + beta*B + eps)
L137-138: X1 = X1 ∘ (Y W.T + beta*W1 A2 R2 + gamma*q3) / (X1 W W.T + beta*X1 + gamma*q4 + eps)
```
* `q1..q4` 是"按 $v_i$ 加权、跨视图累加"的四个统计量，对应 $L_E$ 里 $v_i$ 权重的作用。
* `X1` 的分子三项 = (a) 标签 $YW^\top$ + (c) 锚点路径 $W^cA^fR^f$ + (d) 逐视图还原 $\gamma q_3$ —— 即"全局视图被三方牵引"。

**块 6｜L140–155 目标值**

```python
temp1 = ‖X1 W − Y‖²                     # (a)
temp2 = Σ_i nu[i] ‖x[i] − B p[i]‖²      # (b)
temp3 = ‖B − W1 A1 R1‖² + ‖X1 − W1 A2 R2‖² + ‖A1 − A2‖²    # (c) 含软一致项
temp4 = Σ_i nu[i] ‖X1 d[i] − x[i] s[i]‖²                    # (d)
temp5 = Σ_p ‖W_p‖₂                                          # (e)
objectives = temp1 + alpha*temp2 + beta*temp3 + gamma*temp4 + lamb*temp5
```

**块 7｜L156–195 停机、排序、返回**

```python
if iter == 0: obji = objectives / 2
cver = abs((objectives - obji) / float(obji));  if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)): break
print('the running time of feature selection is {}')      # ← 六篇里唯一会 print 的
w_2 = LA.norm(W, ord=2, axis=1);  f_idx = np.argsort(-w_2)
```
* 排序量是 $W$ 的行 2-范数。⚠️ L192 的注释写"升序排列，最后一个最重要"，**注释错、实现对**（`argsort(-w_2)` 是降序）。

---

## 6. I²VSLC — `alg/I2VSLC.py`（193 行）

### 6.1 骨架表

| 行区间 | 内容 |
|---|---|
| L9 | `eps` |
| L12–26 | 入口与初始化（`w[i]`、`y[i]`、`AA`=标签相关矩阵 C） |
| L28–36 | 视图切分 |
| L38–49 | 每个视图的图 `Sx/Ax/Lx`（**建在特征上**） |
| L51–62 | 迭代准备、`sum_Yt`（各视图标签 OR） |
| L64–76 | 主循环：`w[i]` 更新（**两个对角阵**） |
| L79–81 | 标签相关矩阵 `AA` 更新 |
| L84–95 | `y[i]` 更新（intra + inter） |
| L96–103 | `y[i]` 硬二值化 |
| L105–119 | 目标值前两项 |
| L122–148 | 增强标签 `t1` 与第三项 |
| L150–168 | 第四/五项、停机 |
| L170–192 | 排序、记录、返回 |

### 6.2 逐块解读

**块 0｜L12–26 初始化**

```python
L21-24: for i in range(n_view):
            w.append(rng.random((m[i], label_num)))
            y.append(rng.integers(2, size=(num, label_num)).astype(float))
L26: AA = np.diag([1] * label_num)      # ← 标签相关矩阵 C 初始化为单位阵
```
* `AA` 就是论文的 $C\in\mathbb{R}^{l\times l}$（**全局共享**，不是每视图一个）—— 这是 §3.6.1 澄清框强调的那点。
* 初始化为 $I$ 的语义是"一开始假设标签之间不相关"。

**块 1｜L38–49 视图特征图**

```python
options = {'k': 5, 'weight_mode':'heat_kernel', 't': 1.0, ...}
for i in range(len(m)):
    Sx = dense(construct_W(v[i], **options)); Ax = diag(Σ Sx); Lx_lst.append(Ax - Sx)
```
* 与论文式(3) 一致：**图建在视图特征空间**（"视图特有"的唯一来源），正则作用在标签上。
* `t=1.0` 同样是写死的带宽（§1.4 的坑）。

**块 2｜L57–62 各视图标签的 OR**

```python
sum_Y = y[0].copy()
for i in range(1, n_view): sum_Y += y[i]
sum_Yt = 1[sum_Y > 0]        # Y_vs = 1[Σ_i y⁽ⁱ⁾ ≥ 1]
```
* 对应论文的 $Y_{vs}$（观测标签 = 各视图标签的并集）。

**块 3｜L64–76 `w[i]` 更新：两个对角阵**

```python
L67-69: d_temp = np.full((m[i], 1), LA.norm(w[i], 'fro'))   # ← 一个标量（视图级）
        D = np.diag((0.5 / d_temp).flat)
L71-73: c_temp = np.sqrt(np.sum(np.multiply(w[i], w[i]), 1) + eps)   # ← 逐行（特征级）
        C = np.diag((0.5 / c_temp).flat)
L75-76: w[i] = w[i] ∘ (v[i].T Y) / (v[i].T v[i] w[i] + gamma*D w[i] + lamb*C w[i] + eps)
```
* **这是"层级特征选择"落到代码的样子**：`D` 作用于整个 $w^{(i)}$（视图级组收缩，$\gamma$ 项），`C` 逐行重加权（特征级行稀疏，$\delta$ 项）。
* ⚠️ 分子用的是观测 `Y` 而不是 `y[i]` —— 这是**论文自身的笔误被代码沿用**（论文式(19)(20)(21) 用 $Y$，而式(14)(18) 的拟合项是 $y^{(i)}$）。

**块 4｜L79–81 标签相关矩阵**

```python
AA = AA ∘ (Y.T sum_Yt + Y.T Y) / (2 * (Y.T Y) AA + eps)      # 式(24)
```
* 分子两项 = 与"各视图并集"一致 + 与观测标签一致；分母是当前值的两倍。
* 仓库实现的是 **KKT 正确版**；论文印刷版式(23) 把 $Y^\top Y$ 的符号写反了（METHOD §5.2 M 行）。

**块 5｜L84–95 `y[i]` 更新（intra + inter）**

```python
tem1 = 1[sum_Yt − y[i] ≠ 0]      # 二值化后的指示器
tem2 = 1[y[i] − sum_Yt ≠ 0]      # ← 与 tem1 完全相同！
y[i] = y[i] ∘ (alpha*Sx[i] y[i] + alpha*(n_view-1)*tem1 + v[i] w[i] + beta*(Y AA + tem2))
              / (alpha*Ax[i] y[i] + alpha*(n_view-1)*y[i] + y[i] + beta*y[i] + eps)
```
* 分子四项 = 视图内平滑（$\alpha S^{(i)}y^{(i)}$）+ 与其他视图一致（inter）+ 拟合特征 + 与增强标签一致。
* ⚠️ **`tem1 ≡ tem2`**：论文的 $F=Y_{vs}-y^{(i)}$ 与 $G=y^{(i)}-Y_{vs}$ 是**有符号的连续差**，代码把两者都二值化成同一个 0/1 掩码 ⇒ inter-view 项退化为"对不一致位置统一加常数拉力"，丢掉幅值信息；实测两种口径的 $\Phi$ 相差 **342%**（METHOD §5.2 H 行）。

**块 6｜L96–103 硬二值化**

```python
for i in range(n_view):
    y[i] = MinMax(y[i]); y[i][y[i] <= 0.5] = 0; y[i][y[i] > 0.5] = 1
```
* 论文只对 $Y_{vs}$ 用 $\sum\ge1$ 的绝对阈值，**没有**对 $y^{(i)}$ 做 MinMax（METHOD §5.2 H 行）。

**块 7｜L105–119 目标值前两项**

```python
temp1 = Σ_i ‖v[i] w[i] − y[i]‖²                       # ① 拟合
temp2 = Σ_i Tr(y[i].T Lx[i] y[i])                     # ②a intra
        for j in range(i+1, n_view):
            temp2 += ‖y[i] − y[j]‖²                   # ②b inter（只算 j>i）
if np.isnan(temp2): print("temp2-nan"); break
```
* ⚠️ inter 项只累加 $j>i$（**全对的一半**），而更新式用的是 $n_{view}-1$（按全对计）⇒ **记录的 $\alpha$ 与实际驱动的 $\alpha$ 差 2 倍**（METHOD §5.2 M 行）。
* `temp2` 出现 NaN 时打印并 `break`（论文无此逻辑，属于数值加固）。

**块 8｜L122–148 增强标签与第三项**

```python
t1 = MinMax(Y @ AA);  t1[t1 <= 0.5] = 0; t1[t1 > 0.5] = 1      # 增强标签 YC 的二值化
sum_Y = Σ y[i];  sum_Yt = 1[sum_Y > 0]                        # 重算 OR
tem31 = 1[t1 − sum_Yt ≠ 0];  temp3  = ‖tem31‖²
tem32 = 1[Y − t1 ≠ 0];       temp3 += ‖tem32‖²                # ③ 双向一致
```
* 对应论文式(14) 的 $\|YC-Y_{vs}\|^2+\|Y-YC\|^2$，但代码用**二值化后的 0/1 指示**而不是 Frobenius 差（METHOD §5.2 M 行）。

**块 9｜L150–168 第四/五项与停机**

```python
temp4 = Σ_i ‖w[i]‖_F              # ④ 视图级组收缩
B = concat(w, axis=0);  temp5 = Σ_p ‖B_p‖₂    # ⑤ 特征级行稀疏
objectives = temp1 + alpha*temp2 + beta*temp3 + gamma*temp4 + lamb*temp5
if iter == 0: obji = objectives / 2
cver = abs((objectives - obji) / float(obji))
if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)): break
```
* 停机与其它篇同构（$10^{-3}$、上限 500）。

**块 10｜L170–192 排序与返回**

```python
B = concat(w, axis=0)
w_2 = LA.norm(B, ord=2, axis=1);  f_idx = np.argsort(-w_2)
```
* 排序量 = 拼接权重的行 2-范数。

---

## 附：六篇的"共同套路"对照（按代码顺序）

把六份代码的骨架并排放，你会发现它们只在少数几个位置不同：

| 环节 | TOCL | UGRFS | EF²FS | DHLI | GRAFS | I²VSLC |
|---|---|---|---|---|---|---|
| 每视图权重 | 共享一个 `W` + 选择矩阵 `d[i]` | 每视图 `w[i]` | 每视图 `w[i]` | 每视图 `w[i]`+`u[i]` | 共享 `W`（全局视图上） | 每视图 `w[i]` |
| 额外可学变量 | `y[i]`、`P`、`Z`、`Y_n`、`U/V` | `c[i]`、`wy[i]` | `V`(=G)、`B`、`A` | `Y_c`、`Y_n` | `B`、`X1`、`W1/A1/A2/R1/R2`、`s/p` | `y[i]`、`AA` |
| 图建在哪 | — | **标签** | 视图特征 | — | **标签**（论文写特征） | 视图特征 |
| 循环内变量顺序 | y → T → W → P → Y_n → U → V | w → c → wy | nu → B → G → A → w | w/u → y → Y_c → Y_n | s/p → W → R/A/W1 → B → X1 | w → C → y |
| 目标值里的"技巧" | `0.5*temp2+wtnn`、`2*beta`、`2*lamb` | 首轮 `/2` | 首轮 `/2` | `2β`、`2γ` | 六项到齐 | `t1` 二值化 |
| 停机阈值 | $10^{-6}$ | $10^{-3}$ | $10^{-3}$ | $10^{-3}$ | $10^{-3}$ | $10^{-3}$ |
| 迭代上限 | 500（常被 `MVML_MAX_ITER` 限流） | 500 | **100** | 500 | 500 | 500 |
| 排序量 | $\|W\|_{2,\cdot}$ | 拼接 $\|W\|_{2,\cdot}$ | $\|A\|_{2,\cdot}$ | $\|{\rm MinMax}(W)+{\rm MinMax}(U)\|_{2,\cdot}$ | $\|W\|_{2,\cdot}$ | 拼接 $\|W\|_{2,\cdot}$ |
| 硬二值化 | `y` | — | — | `y`、`Y_c`、`Y_n` | — | `y` |

> **读代码的固定顺序**：① 初始化（哪些变量、怎么随机）→ ② 每轮第一个动作（谁先更新，往往暴露"用新值还是旧值"）→
> ③ 图/权重这些"外部量"是循环内算还是循环外算 → ④ 目标值里有没有多因子/替换项 → ⑤ 停机阈值 → ⑥ 排序量。
> 六篇的**真正差异全在 ②③④**，其余都是同一套模板。
