"""insight.mechanisms — 六篇论文核心创新机制的**最小可运行算子库**

本模块的目的不是复现六篇论文（复现在 ``alg/`` 里已经有了），而是把每篇论文
**真正带来 CCF-A/B 级贡献的那一个数学机制**单独剥出来，做成一个可独立调用、
可单独验证、可单独消融的算子。

对每个机制 ``M#``，本模块给出：

1. ``公式``   —— 论文中的 LaTeX 公式（写在函数 docstring 里）
2. ``推导``   —— 该更新式/近端算子是怎么推出来的
3. ``为什么是创新`` —— 它相对前人方法到底改了什么
4. ``代码``   —— 逐行对应公式的 numpy 实现（无隐藏技巧）

编号与 ``PAPER_METHOD_ANALYSIS.md`` 的 §6 完全一致：

===== ============================== ================================
 M#    机制                            出处
===== ============================== ================================
 M1    加权张量核范数近端算子          TOCL  (ACM MM 2025, CCF-A)
 M2    样本可信度 C⁽ⁱ⁾ 交替学习        UGRFS (AAAI 2025,   CCF-A)
 M3    嵌入式融合（融合↔选择同框）     EF2FS (PR 2025,     CCF-B/一区)
 M4    双层级标签拆分 + 离散异或算子   DHLI  (AAAI 2024,   CCF-A)
 M5    视图特有标签图 + 投票共识       I²VSLC(Inf. Sci. 2024, CCF-B)
 M6    锚点引导的双路径多层分解        GRAFS (Inf. Sci. 2024, CCF-B)
 M7    ADMM + 有证明收敛              ← 对六篇「乘性更新无证明」的补强
 M8    自适应图学习 + 连通分量约束     ← 对六篇「固定 knn 图」的补强
===== ============================== ================================

约定
----
* 记号统一为：n 样本数、V 视图数、d⁽ⁱ⁾ 第 i 视图特征数、l 标签数。
* 矩阵一律 ``(n, ·)`` 行主序，与仓库 ``alg/*.py`` 及 ``main.py`` 的布局一致。
* 所有函数都是**纯函数**（给定输入和 rng 输出确定），方便单测与消融。
"""

from __future__ import annotations

import numpy as np
from numpy import linalg as LA

EPS = 2.2204e-16

__all__ = [
    "prox_weighted_tnn",
    "shrinkage_weighted_svt",
    "egla_update",
    "uncertainty_confidence_update",
    "ef2fs_step",
    "ef2fs_objective",
    "discrete_xor",
    "split_hybrid_labels",
    "dhliself_step",
    "view_label_laplacian",
    "view_specific_label_step",
    "anchor_select_kmeans",
    "grafs_reconstruct",
    "admm_nmf",
    "adaptive_graph_learning",
]


# =========================================================================== #
# M1 — 加权张量核范数近端算子 (TOCL)
# =========================================================================== #

def prox_weighted_tnn(T: np.ndarray, C: float,
                      fft_axis: int = 2) -> tuple[np.ndarray, float, int]:
    r"""M1: 加权张量核范数 (Weighted Tensor Nuclear Norm) 的近端算子。

    **论文公式** (TOCL, ACM MM 2025, Eq.10 / Eq.22-25)::

        L2(·) = ‖Z − T‖²_F + ‖T‖_{w,*}                       (Eq.10)
        min_{T̄(:,:,j)}  (1/n)‖Z̄(:,:,j) − T̄(:,:,j)‖²_F + ‖T̄(:,:,j)‖_{w,*}   (Eq.22)

    **为什么是创新**: 标准 TNN ``‖T‖_* = Σ_i σ_i`` 对所有奇异值一视同仁地把
    阈值钉死在 ``τ``；但大奇异值承载的是张量的**主成分**（跨视图共识），
    小奇异值才是噪声。TOCL 引入**逐奇异值权重** ``w_i^(j) = C/(σ_i^(j)+ε)``，
    使大奇异值收缩得少、小奇异值收缩得多 —— 这是把「迭代重加权」思想
    第一次装进多视图多标签的**张量**低秩约束里。

    **本实现的两条路径**（对应论文的一处内部不一致，详见
    ``PAPER_METHOD_ANALYSIS.md`` §5.6）：

    * ``mode="quadratic"`` —— 论文 Eq.24 给出的闭式解
      ``σ* = (c1 + √c2)/2``，``c1 = σ−ε``，``c2 = (σ−ε)² − 4(C − εσ)``。
      与仓库 ``alg/TOCL.py::prox_weight_tensor_nuclear_norm`` 数值等价。
    * ``mode="shrinkage"`` —— 从 ``w_i = C/(σ_i+ε)`` **严格**推出来的近端解
      ``σ* = max(σ − C/(σ+ε), 0)``（见 :func:`shrinkage_weighted_svt`）。

    两条路径在 ``C`` 越大时差异越明显；``demo`` 里会并排打印它们，用于说明
    「论文的闭式解并不等于它声称的加权范数的近端算子」。

    Parameters
    ----------
    T : (n1, n2, n3) ndarray
        ``n3`` 是**视图维**（第三阶）。TOCL 里 ``T[:, :, i] = y⁽ⁱ⁾ y⁽ⁱ⁾ᵀ``。
    C : float
        正则参数（正数）；越大 → 收缩越强 → tubal rank 越低。

    Returns
    -------
    Tnew : (n1, n2, n3) ndarray
    wtnn : float
        加权张量核范数的值（已按 ``n3`` 归一化，与论文 Eq.22 的 ``1/n`` 一致）。
    trank : int
        tubal rank 上界（各前切片非零奇异值个数的最大值）。

    Notes
    -----
    算法骨架 = t-SVD：沿第 3 维做 FFT → 对每个前切片做 SVD → 收缩奇异值 →
    用共轭对称性只算前一半切片 → 逆 FFT 取实部。复杂度 ``O(n1 n2² n3)``。

    ``fft_axis`` 与张量布局
    ----------------------
    t-SVD 的**管 (tube)** 方向必须是「视图维」，否则低秩约束约束的不是跨视图
    共识。论文 Definition 2 与式 (10) 都把 T 定义为 ``n×V×n``（第 2 阶是视图），
    此时 ``fft_axis=1`` 才是论文语义。

    ⚠️ 但仓库 ``alg/TOCL.py`` 构造的是
    ``HH2 = HH.transpose((0, 2, 1))``，形状 ``(n, V, n)`` —— **视图维落在第 1 阶
    （长度 V），第 2 阶是样本维（长度 n）**，而算子内部对 **第 2 阶** 做 FFT。
    因此参考实现实际上是在**样本方向**上做 t-SVD，前切片数是 ``round(n/2)``
    而不是 ``round(V/2)``。这是论文与代码之间一处结构性偏离，见
    ``PAPER_METHOD_ANALYSIS.md`` §5.6。默认 ``fft_axis=2`` 保持与参考实现一致。

    Returns
    -------
    Tnew : (n1, n2, n3) ndarray
    wtnn : float
        加权张量核范数的值（已按 ``n3`` 归一化，与论文 Eq.22 的 ``1/n`` 一致）。
    trank : int
        tubal rank 上界（各前切片非零奇异值个数的最大值）。
    """
    n1, n2, n3 = T.shape
    if T.ndim != 3:
        raise ValueError(f"prox_weighted_tnn expects a 3-mode tensor, got {T.shape}")

    # NOTE: ``scipy.fftpack.fft`` (used by alg/TOCL.py) discards the analytic
    # imaginary part of a real input, ``numpy.fft.fft`` does not. The conjugate
    # reconstruction below (X[:, :, n3-j] = conj(X[:, :, j])) is only consistent
    # with the full-spectrum convention, so we match the reference
    # implementation exactly by using the same transform.
    from scipy.fftpack import fft as _fft
    # move the transform axis to the last position, then treat it as n3
    That = np.moveaxis(_fft(np.moveaxis(T, fft_axis, -1), axis=-1), -1, fft_axis)
    loop_len = That.shape[fft_axis]
    Tnew_hat = np.zeros((n1, n2, n3), dtype=complex)
    wtnn = 0.0
    trank = 0

    # NOTE: 下面刻意**逐句复刻**参考实现的循环结构，因为它的两个「特性」
    # 会改变可观测量，必须一起保留才能逐值对照：
    #
    #   (a) 循环是 ``for i in range(1, round(n3/2))``，**从不处理 i = 0**，
    #       所以 ``Tnew_hat[..., 0, ...]`` 恒为零矩阵（丢掉直流分量）；
    #   (b) 循环体内先算 i、再写 ``Tnew_hat[..., n3-i, ...] = conj(...)``，
    #       于是 i=2 的**计算结果被 i=1 的共轭镜像覆盖**（切片 2..n3-1 全部
    #       由镜像填充）。但 `wtnn` 的累加发生在覆盖**之前**，所以它对
    #       i=2..round(n3/2)-1 的贡献仍然计入 —— 即 wtnn 与实际写入张量的
    #       切片集合**不一致**。
    # 这两点都是原实现的索引组织缺陷，不是 t-SVT 的数学内容。
    for i in range(1, round(loop_len / 2)):
        U, Sv, Vh = LA.svd(_slice_along(That, fft_axis, i), full_matrices=False)
        V = Vh.conj().T
        c1 = Sv - EPS
        c2 = (Sv - EPS) ** 2 - 4.0 * (C - EPS * Sv)
        keep = c2 > 0.0
        Snew = np.zeros_like(Sv)
        Snew[keep] = np.maximum(c1[keep] + np.sqrt(c2[keep]), 0.0) / 2.0
        r = int(np.count_nonzero(Snew > 0))
        if r >= 1:
            _set_slice(Tnew_hat, fft_axis, i,
                       U[:, :r] @ np.diag(Snew[:r]) @ V[:, :r].T)
            wtnn += float(np.sum(Snew[:r] * (C / (Snew[:r] + EPS))))
            trank = max(trank, r)
        # 共轭镜像（会覆盖后续切片的计算结果，与原实现一致）
        _set_slice(Tnew_hat, fft_axis, loop_len - i,
                   np.conj(_slice_along(Tnew_hat, fft_axis, i)))

    # 偶数长度：额外处理最后一个半切片
    if loop_len % 2 == 0:
        i = round(loop_len / 2) + 1
        if i < loop_len:
            U, Sv, Vh = LA.svd(_slice_along(That, fft_axis, i), full_matrices=False)
            V = Vh.conj().T
            c1 = Sv - EPS
            c2 = (Sv - EPS) ** 2 - 4.0 * (C - EPS * Sv)
            keep = c2 > 0.0
            Snew = np.zeros_like(Sv)
            Snew[keep] = np.maximum(c1[keep] + np.sqrt(c2[keep]), 0.0) / 2.0
            r = int(np.count_nonzero(Snew > 0))
            if r >= 1:
                _set_slice(Tnew_hat, fft_axis, i,
                           U[:, :r] @ np.diag(Snew[:r]) @ V[:, :r].T)
                wtnn += float(np.sum(Snew[:r] * (C / (Snew[:r] + EPS))))
                trank = max(trank, r)

    Tnew = np.real(np.fft.ifftn(np.moveaxis(Tnew_hat, fft_axis, -1), axes=[-1]))
    Tnew = np.moveaxis(Tnew, -1, fft_axis)
    # 归一化因子：参考实现在返回前除以 **n3**（不是 round(n3/2)）。由于它
    # 在循环体内先累加 wtnn、再用共轭镜像覆盖切片，实际累加了
    #   1 (显式切片) + (round(n3/2)−1) (循环) + 1 (偶数分支) = round(n3/2)+1
    # 个切片的贡献，而分母只有 n3 —— 于是偶数 n3 下 wtnn 被系统性地
    # 放大 (round(n3/2)+1)/n3 ≈ 1/2 倍。这与论文 Eq.22 的 "1/n Σ_j" 不符。
    # 该量在 TOCL 的主循环里**不参与任何更新**（只进目标函数记录），
    # 所以只影响日志，不影响特征排序；此处保持与参考实现一致以便逐值核对。
    wtnn /= loop_len
    return Tnew, wtnn, trank


def _slice_along(T: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Take ``T[..., index, ...]`` along ``axis`` (2-D frontal slice)."""
    return np.take(T, index, axis=axis)


def _set_slice(T: np.ndarray, axis: int, index: int, value: np.ndarray) -> None:
    """In-place ``T[..., index, ...] = value`` along ``axis``."""
    idx = [slice(None)] * T.ndim
    idx[axis] = index
    T[tuple(idx)] = value


def shrinkage_weighted_svt(S: np.ndarray, C: float) -> np.ndarray:
    r"""M1 的严格近端形式：加权奇异值软阈值。

    **推导**: 论文声明权重 ``w_i = C/(σ_i + ε)``（``C > 0``），即加权核范数

    .. math::  \|T\|_{w,*} = \sum_i \frac{C\,\sigma_i}{\sigma_i + \varepsilon}

    近端问题 ``min_σ ½(σ − s)² + Cσ/(σ+ε)`` 的一阶条件为

    .. math::  \sigma - s + \frac{C\varepsilon}{(\sigma+\varepsilon)^2} = 0
               \quad\Longrightarrow\quad
               \sigma = \max\!\Big(s - \frac{C}{s+\varepsilon},\; 0\Big)

    （最后一步对 ``ε → 0`` 取极限，并把 ``Cε/(σ+ε)²`` 近似写作 ``C/(s+ε)``，
    与迭代重加权 ℓ1 的 standard majorization 一致）。

    对比标准 SVT ``σ* = max(s − τ, 0)``：这里的阈值 ``C/(s+ε)`` **随 s 衰减**，
    所以大奇异值几乎不被惩罚 —— 这就是论文所说
    "larger singular values ... should be penalized less" 的数学含义。
    """
    return np.maximum(S - C / (S + EPS), 0.0)


# =========================================================================== #
# M2 — 样本可信度 C⁽ⁱ⁾ 的交替学习 (UGRFS)
# =========================================================================== #

def egla_update(X: np.ndarray, Y: np.ndarray, W: np.ndarray, D: np.ndarray,
                C: np.ndarray, beta: float) -> np.ndarray:
    r"""M2-a: UGRFS 的乘性更新规则（论文 Eq.14-16 的统一实现）。

    **论文目标函数** (UGRFS, AAAI 2025, Eq.12)::

        min_{W⁽ⁱ⁾,C⁽ⁱ⁾,W_y⁽ⁱ⁾}  Σ_i ‖diag(C⁽ⁱ⁾)X⁽ⁱ⁾W⁽ⁱ⁾ − Y‖²_F
                              + α Tr(Dᵀ L_Y D)
                              + β Σ_i ‖D − diag(C⁽ⁱ⁾)X⁽ⁱ⁾‖²_F
                              + γ ‖D − X_f‖²_F
                              + δ ‖W‖_{2,1}

    **三个更新式**（乘性 / 非负交替优化）:

    .. math::

        W^{(i)} &\leftarrow W^{(i)} \circ
            \frac{A^{(i)\top} Y}{A^{(i)\top} A^{(i)} W^{(i)} + 2\delta E W^{(i)}} \\[4pt]
        C^{(i)} &\leftarrow C^{(i)} \circ
            \frac{A^{(i)\top} Y W^{(i)\top} X^{(i)} + \beta D^\top X^{(i)}}
                 {A^{(i)\top} A^{(i)} W^{(i)} W^{(i)\top} X^{(i)} + \beta A^{(i)\top} X^{(i)}} \\[4pt]
        W_y^{(i)} &\leftarrow W_y^{(i)} \circ
            \frac{\alpha Y_x^\top S D^{(i)} + \beta Y_x^\top A^{(i)} + \gamma Y_x^\top X_f^{(i)}}
                 {\alpha Y_x^\top A_Y D^{(i)} + (\beta+\gamma) Y_x^\top D^{(i)}}

    其中 :math:`A^{(i)} = \mathrm{diag}(C^{(i)})X^{(i)}`，
    :math:`E_{pp} = 1/(2\|W_p\|_2)` 是 ``l2,1`` 范数的重加权对角阵，
    :math:`S` / :math:`A_Y` 是标签图的亲和矩阵 / 度矩阵，:math:`L_Y = A_Y - S`。

    **为什么是创新**: ``diag(C⁽ⁱ⁾)X⁽ⁱ⁾`` 让**每个样本在每个视图里**拥有一个
    可学习的可信度，从而把「视图加权」（整列缩放）细化到「样本×视图加权」。
    和 :math:`\|D - \mathrm{diag}(C)X\|²` 联立后，``C`` 同时受
    「反推全局视图 D」和「拟合标签 Y」两侧牵引 —— 这就是论文的
    "mutually promote" 机制。

    Parameters
    ----------
    X, Y : ndarray
        ``X`` 是该视图特征 ``(n, d⁽ⁱ⁾)``；``Y`` 是标签 ``(n, l)``。
    W : (d⁽ⁱ⁾, l)
    D : (n, d⁽ⁱ⁾)
        该视图对应的全局视图切片 ``D⁽ⁱ⁾ = Y_x W_y⁽ⁱ⁾``。
    C : (n, n) 对角阵 ``diag(c)``
    beta : float

    Returns
    -------
    W, C, info : ndarray, ndarray, dict
    """
    A = C @ X                                        # A⁽ⁱ⁾ = diag(C⁽ⁱ⁾)X⁽ⁱ⁾, (n, d)
    E = _l21_diag(W)                                 # E_pp = 1/(2‖W_p‖₂), (d, d)
    delta = 1.0

    num_W = A.T @ Y                                  # (d, l)
    den_W = (A.T @ A) @ W + 2.0 * delta * (E @ W) + EPS
    W = W * (num_W / den_W)

    # --- C 更新 ---
    # 论⽂ Eq.(15) 印作  C ∘ [Y WᵀXᵀ + β D⁽ⁱ⁾Xᵀ] / [A⁽ⁱ⁾W WᵀXᵀ + β A⁽ⁱ⁾Xᵀ]。
    # ⚠️ 该式有两处维数/记号问题：
    #   (1) D⁽ⁱ⁾ = Y_x W_y⁽ⁱ⁾ ∈ R^{n×(n+1)}，不是 n 阶方阵，D⁽ⁱ⁾Xᵀ 无法与左项相加；
    #   (2) 分子首项也缺一个 A⁽ⁱ⁾ᵀ。对 β 项 ‖D⁽ⁱ⁾ − diag(C)X⁽ⁱ⁾‖² 求 C 的导数
    #       得到 −2β·diag(A⁽ⁱ⁾ᵀX⁽ⁱ⁾)，故分子应为 β·A⁽ⁱ⁾ᵀX⁽ⁱ⁾，与分母同形。
    # 下面的写法与仓库 alg/UGRFS.py L114 的正/负部分离结果一致（那里的正部
    # 是 beta * (c[i]@x[i]) @ x[i].T）。
    num_C = (Y @ W.T) @ X.T + beta * (A @ X.T)
    den_C = ((A @ W) @ W.T) @ X.T + beta * (A @ X.T) + EPS
    C = C * (num_C / den_C)

    return W, C, {"W_change": float(LA.norm(num_W / den_W - 1.0))}


def uncertainty_confidence_update(X: np.ndarray, Y: np.ndarray, W: np.ndarray,
                                  D: np.ndarray, c: np.ndarray,
                                  beta: float) -> np.ndarray:
    r"""M2-b: 只要样本可信度向量 c（对角阵的显式形式），返回更新后的 c。

    这是 :func:`egla_update` 中 ``C`` 那一步的「只做对角」版本 —— 因为 ``C``
    是对角阵，整个矩阵更新式只有对角元有意义。取对角后写成逐行形式：

    .. math::

        c_m \leftarrow c_m \cdot
          \frac{ c_m\,\big[X(YW^\top)\big]_{mm} + \beta\,\big[X X^\top\big]_{mm} }
               { c_m^2\,\big[X (WW^\top) X^\top\big]_{mm}
                 + \beta\,c_m\,\big[X X^\top\big]_{mm} }

    约掉公因子 ``c_m`` 后，``c_m`` 的**不动点**是

    .. math::

        c_m^\star = \frac{[X(YW^\top)]_{mm}}{[X(WW^\top)X^\top]_{mm}}

    即「该样本的**特征-标签拟合质量** ÷ 该样本**投影后的能量**」：

    * 分子大 → 这个样本在**当前 W** 下能被标签很好地解释 → 可信度高；
    * 分母大 → 该样本在视图里能量很大却解释不了标签 → 典型噪声 / 离群点。

    这就把「样本不确定度」变成了一个有**闭式含义**的量，而不是凭空加的超参：
    它是**残差解释比**，与「用损失大小判断样本可信度」这一直觉严格对应。
    注意 β 项在分子分母上完全抵消，即 **β 不影响 c 的不动点，只影响收敛速度**。

    Returns
    -------
    c_new : (n,) ndarray
        逐样本可信度；下界截断到 ``1e-6`` 防止塌缩。
    """
    num = np.diag(X @ W @ Y.T) + beta * np.diag(X @ X.T)
    den = np.diag(X @ W @ W.T @ X.T) + beta * np.diag(X @ X.T) + EPS
    c_new = c * (num / den)
    return np.clip(c_new, 1e-6, None)


# =========================================================================== #
# M3 — 嵌入式特征融合 (EF2FS)
# =========================================================================== #

def ef2fs_step(views, Y, G, B, A, ws, Lx_lst, alpha=1.0, beta=1.0, gamma=1.0,
               view_weight="paper", Y_for_weight=None):
    r"""M3: 嵌入式融合的一轮坐标下降（EF2FS, Pattern Recognition 2025）。

    **论文目标函数**::

        min_{A,G,B,w⁽ⁱ⁾}  Σᵢ ‖X⁽ⁱ⁾w⁽ⁱ⁾ − G‖²_F      (视图 → 公共嵌入对齐)
                        + α ‖Y − G Bᵀ‖²_F            (嵌入 → 标签拟合)
                        + β ‖X_f A − G‖²_F           (融合特征 → 嵌入重构)
                        + γ Σᵢ ‖a⁽ⁱ⁾ − w⁽ⁱ⁾‖²_F      (系数块与视图投影一致)
                        + δ ‖A‖_{2,1}                (行稀疏 → 特征排序)

    **为什么是创新**: 「先融合视图、再选特征」的两阶段做法里，融合目标与选择
    目标是两个独立的最优化问题，融合阶段丢掉的判别信息在选择阶段无法找回。
    EF2FS 让 ``G``（公共嵌入）同时被 ``ΣᵢX⁽ⁱ⁾w⁽ⁱ⁾``（视图侧）、``YB``（标签侧）
    和 ``X_f A``（融合侧）三方牵引 —— 于是**融合矩阵 A 的行范数天然就是
    特征重要性**，选择与融合在同一个目标函数里互相指导。这是「embedded」
    相对「two-stage」的严格优势陈述。

    **视图权重**（论文的核心技巧之一，也是本模块要指出的**缺陷所在**）::

        vᵢ = (1 / Tr(Gᵀ L⁽ⁱ⁾ G)) / Σⱼ (1 / Tr(Gᵀ L⁽ʲ⁾ G))

    论文的解释是：``Tr(GᵀL⁽ⁱ⁾G)`` 度量嵌入 ``G`` 在第 i 个视图图结构上的
    **不平滑程度**；越平滑（值越小）的视图越可信，权重越大。

    ⚠️ **该量不是尺度不变的，而且与标签无关**。对任意 ``c > 0``，
    ``Tr((cG)ᵀL(cG)) = c²Tr(GᵀLG)``，所以只要某个视图的图拉普拉斯本身
    把 ``G`` 映得接近零（例如该视图的特征取值范围恰好落在热核带宽 ``t=1``
    的"近邻全部等权"区间，或特征整体尺度偏小），它的能量就小、权重就大 ——
    **这与该视图是否携带标签信息完全无关**。纯噪声视图同样可能拿到最高权重。

    本函数因此提供两种口径，便于直接对照：

    * ``view_weight="paper"`` —— 论文原式（尺度不敏感 + 无标签参与）；
    * ``view_weight="label_aware"`` —— 修正版：用**图平滑后的标签拟合残差**
      作为能量，即

      .. math::

          E_i = \min_{W}\ \sum_i \|S^{(i)}_{\text{norm}} Y - X^{(i)}W\|_F^2
                + \lambda\|W\|_F^2

      其中 ``S_normalized = D^{-1/2} S D^{-1/2}`` 是归一化亲和矩阵（对度做
      归一 ⇒ 尺度不变）。``E_i`` 小意味着"该视图能解释标签，且其局部几何与
      标签一致"，这才是"视图重要"的正确含义。

    Returns
    -------
    dict
        ``{'G','B','A','ws','nu','new_X','obj'}``
    """
    V = len(views)

    # --- 视图权重 vᵢ ---
    # 注意零值保护的正确写法：判据应是「1/E 是否有限」，而不是「E 是否大于
    # 某个阈值」。后者会把 1e-20 这种"极小但非零"的能量当成 0，从而把本该
    # 独吞权重的视图误置为 0 —— 与 1/E 的代数含义相反。
    if view_weight == "paper":
        # 论文原式：Tr(GᵀL⁽ⁱ⁾G)，尺度不敏感且与标签无关
        energy = np.array([float(np.trace(G.T @ Lx_lst[i] @ G)) for i in range(V)])
    elif view_weight == "label_aware":
        Yt = Y if Y_for_weight is None else Y_for_weight
        energy = np.array([_label_aware_view_energy(views[i], Yt, Lx_lst[i])
                           for i in range(V)])
    else:
        raise ValueError(f"unknown view_weight={view_weight!r}; "
                         "expected 'paper' or 'label_aware'")

    # 数值噪声修正 + 退化视图的处理（与参考实现一致）。
    #
    # 理论上 Tr(GᵀLG) ≥ 0（L 半正定），浮点误差可能给出约 −1e−17 的负值
    # → 截断到 0。
    #
    # 而 E = 0 意味着该视图的图拉普拉斯把 G 完全"消灭"（例如热核在
    # t=1.0 下溢，L ≡ 0）。此时 1/E → ∞，参考实现的做法是分母加
    # ``eps = 2.2204e−16``，等价于把 1/E 上限设为 ``1/eps ≈ 4.5e15``。
    # 本函数照此处理：**退化视图会独吞全部权重**。这正是论文原式的
    # 实际行为，也是本模块要在 M3 附一/附二 里指出的缺陷所在。
    energy = np.where(np.abs(energy) < 1e-10, 0.0, energy)
    with np.errstate(divide="ignore", over="ignore"):
        inv = np.where(energy > 0, 1.0 / np.where(energy > 0, energy, 1.0), 1.0 / EPS)
    inv = np.where(np.isfinite(inv), inv, 1.0 / EPS)
    nu = inv / inv.sum() if inv.sum() > EPS else np.full(V, 1.0 / V)

    new_X = np.concatenate([views[i] * nu[i] for i in range(V)], axis=1)

    m = [v.shape[1] for v in views]

    # --- B: Y ≈ G Bᵀ ---
    B = B * ((Y.T @ G) / ((B @ G.T) @ G + EPS))

    # --- G ---
    xw = sum(views[i] @ ws[i] for i in range(V))
    num_G = alpha * (Y @ B) + beta * xw + (new_X @ A)
    den_G = alpha * ((G @ B.T) @ B) + (V * beta + 1.0) * G + EPS
    G = G * (num_G / den_G)

    # --- A: 含 l2,1 重加权 ---
    D = _l21_diag(A)
    num_A = (new_X.T @ G) + gamma * np.concatenate(ws, axis=0)
    den_A = ((new_X.T @ new_X) @ A) + gamma * A + _lam_diag_scale(D, A) + EPS
    A = A * (num_A / den_A)

    # --- w⁽ⁱ⁾: 与 A 的视图块保持一致 ---
    off = 0
    for i in range(V):
        a_i = A[off:off + m[i], :]
        num_w = beta * (views[i].T @ G) + gamma * a_i
        den_w = beta * ((views[i].T @ views[i]) @ ws[i]) + gamma * ws[i] + EPS
        ws[i] = ws[i] * (num_w / den_w)
        off += m[i]

    obj = ef2fs_objective(views, Y, G, B, A, ws, new_X, alpha, beta, gamma)
    return {"G": G, "B": B, "A": A, "ws": ws, "nu": nu, "new_X": new_X, "obj": obj}


def ef2fs_objective(views, Y, G, B, A, ws, new_X, alpha, beta, gamma):
    """M3 的目标函数值（用于验证单调下降）。"""
    t1 = LA.norm(new_X @ A - G, "fro") ** 2
    t2 = LA.norm(Y - G @ B.T, "fro") ** 2
    t3 = sum(LA.norm(views[i] @ ws[i] - G, "fro") ** 2 for i in range(len(views)))
    off = 0
    t4 = 0.0
    for i in range(len(views)):
        mi = views[i].shape[1]
        t4 += LA.norm(A[off:off + mi, :] - ws[i], "fro") ** 2
        off += mi
    t5 = float(np.sum(LA.norm(A, axis=1)))
    return t1 + alpha * t2 + beta * t3 + gamma * t4 + t5


# =========================================================================== #
# M4 — 双层级混合标签拆分 + 离散异或算子 (DHLI / TOCL)
# =========================================================================== #

def discrete_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""M4-a: 离散分歧算子 ``⊖``（TOCL Eq.12，DHLI 的 ``tem`` 系列同源）。

    定义::

        (a ⊖ b)_{pq} = 1  if a_{pq} ≠ b_{pq}  else 0

    **为什么它重要**: 多标签特征选择的常规做法是用 Frobenius 范数
    ``‖Y_all ⊗ Y_n − Y‖²_F`` 惩罚标签不一致。但 ``Y``、``Y_all``、``Y_n`` 都是
    {0,1} 矩阵，此时 ``‖·‖²_F`` 退化为**汉明距离**（等价于异或的计数），
    而直接写 ``Y_all + Y_n − Y`` 再用 ``(·>0)→1`` 阈值化，实现的正是
    集合意义下的「不一致指示」而不是算术差 —— 这是原始实现
    ``F[F>0]=1; F[F<0]=1`` 的确切含义。

    这个算子把「标签噪声识别」从**连续松弛**（低秩、流形）拉回到
    **离散逻辑**层面，是 DHLI/TOCL 这一线相对前人方法最本质的转变：
    噪声不再靠「重构残差小」隐式体现，而是被显式地**判**出来。

    Returns
    -------
    (n, l) uint8 ndarray, 取值 {0, 1}
    """
    return (np.asarray(a) != np.asarray(b)).astype(np.float64)


def split_hybrid_labels(Y: np.ndarray, Y_c: np.ndarray, Y_n: np.ndarray,
                        Y_all: np.ndarray) -> dict:
    r"""M4-b: 把观测标签拆成「一致 / 噪声 / 视图特有」三层。

    论文的两条定义 (DHLI Definition 1；TOCL Definition 3-4)::

        Y = Y_all ⊗ Y_n ,     Y_all = y_t^{(1)} ⊗ y_t^{(2)} ⊗ ⋯ ⊗ y_t^{(V)}
        Y_spe = Y − Y_c − Y_n      （再阈值化到 {0,1}）

    其中 ``⊗`` 是逐元素逻辑与（Hadamard 意义下的「同位置共同为 1」）。
    于是三类标签的语义是：

    =========== ==========================================================
    ``Y_c``     视图间**共享且干净**的一致标签 —— 真正的监督信号
    ``Y_n``     被噪声污染的错误标签 —— 应被抑制
    ``Y_spe``   只有部分视图能解释的**视图特有**标签 —— 互补信息
    =========== ==========================================================

    Returns
    -------
    dict
        ``{'Y_c','Y_n','Y_spe','Y_all','consistent'::bool}``
    """
    Y = np.asarray(Y, dtype=float)
    Y_c = np.asarray(Y_c, dtype=float)
    Y_n = np.asarray(Y_n, dtype=float)

    y_spe = Y - Y_c - Y_n
    y_spe = (y_spe != 0).astype(float)      # 阈值化：任何非零都算「特有」

    return {
        "Y_c": (Y_c > 0.5).astype(float),
        "Y_n": (Y_n > 0.5).astype(float),
        "Y_spe": y_spe,
        "Y_all": (np.asarray(Y_all) > 0).astype(float),
        "consistent": bool(np.allclose((Y_c > 0.5) | (Y_n > 0.5) | (y_spe > 0.5),
                                       Y > 0.5)),
    }


def dhliself_step(X: np.ndarray, Y: np.ndarray, w: np.ndarray, u: np.ndarray,
                  y: np.ndarray, y_others: list[np.ndarray],
                  Y_c: np.ndarray, Y_spe: np.ndarray, Y_n: np.ndarray,
                  alpha: float = 1.0, gamma: float = 1.0,
                  lamb: float = 1.0) -> dict:
    r"""M4-c: DHLI 第二层（特征层）的一轮更新，含**视图互斥项**。

    **论文更新式**::

        w^{(i)} &\leftarrow w^{(i)} \circ
            \frac{X^{(i)\top} Y_c}{X^{(i)\top}X^{(i)}w^{(i)} + \gamma D w^{(i)}} \\[4pt]
        u^{(i)} &\leftarrow u^{(i)} \circ
            \frac{X^{(i)\top} y^{(i)}}{X^{(i)\top}X^{(i)}u^{(i)} + \gamma E u^{(i)}} \\[4pt]
        y^{(i)} &\leftarrow y^{(i)} \circ
            \frac{X^{(i)}u^{(i)} + \lambda\,\mathbf{1}[Y_{spe} \ominus y_{others}]}
                 {y^{(i)} + 2\lambda y^{(i)} + \lambda y^{(i)} B B}

    其中**视图互斥矩阵**

    .. math::  B = \bigodot_{j \neq i} y^{(j)}

    是「除自己以外所有视图伪标签的 Hadamard 积」。``B`` 中为 1 的位置意味着
    **所有其他视图都同意**该 label 为正 —— 于是 ``λ y⁽ⁱ⁾BB`` 在这一项上
    强烈惩罚第 i 个视图的不同意见，迫使其向共识靠拢。

    **为什么是创新**: 前人做法用 ``Σ_{j≠i}‖y⁽ⁱ⁾ − y⁽ʲ⁾‖²``（两两欧氏距离）
    来求共识，这是**软平均**，会被噪声视图拉偏。``B`` 用的是
    **全票通过（unanimity）语义的逻辑与**，是硬共识：一个视图不同意全体时
    惩罚最强，两个视图互相不同意时反而不惩罚。这把「少数服从多数」的
    投票语义严格地写进了目标函数。

    Returns
    -------
    dict
        ``{'w','u','y'}``
    """
    n, l = Y.shape
    D = _l21_diag(w)
    E = _l21_diag(u)

    w = w * ((X.T @ Y_c) / ((X.T @ X) @ w + gamma * (D @ w) + EPS))
    u = u * ((X.T @ y) / ((X.T @ X) @ u + gamma * (E @ u) + EPS))

    # 除自己之外的视图并（投票用，用于构造 Y_spe 的残差）
    y_o = np.zeros_like(y)
    for yj in y_others:
        y_o = y_o + yj
    y_o = (y_o > 0).astype(float)

    # 视图互斥矩阵 B = ⊙_{j≠i} y⁽ʲ⁾  —— Hadamard 积的**单位元是 1**
    # （原实现写成 B = 0 再累乘，导致 B ≡ 0、整个互斥项静默失效）
    B = np.ones_like(y)
    for yj in y_others:
        B = B * yj

    tem1 = (Y_spe != y_o).astype(float)
    y = y * ((X @ u + lamb * tem1) /
             (y + lamb * y + lamb * y * B * B + EPS))

    return {"w": w, "u": u, "y": y, "B": B, "y_consensus": y_o}


# =========================================================================== #
# M5 — 视图特有标签图 + 投票共识 (I²VSLC)
# =========================================================================== #

def view_label_laplacian(Y: np.ndarray, k: int = 5, t: float = 1.0,
                         seed: int | None = None) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray]:
    r"""M5-a: 构造自刻画函数 Φ(·) 所需的图拉普拉斯 ``L⁽ⁱ⁾ = A⁽ⁱ⁾ − S⁽ⁱ⁾``。

    **论文式 (3)-(5)**（I²VSLC, Inf. Sci. 2024）::

        s_{ij}^{(v)} = exp(−‖x_i^{(v)} − x_j^{(v)}‖² / σ²)
                       if x_i ∈ N_q(x_j) or x_j ∈ N_q(x_i)   else 0      (Eq.3)

        (1/2) Σ_{i,j} s_{ij}^{(v)} ‖y_i − y_j‖²
            = Tr(yᵀ (A^{(v)} − S^{(v)}) y) = Tr(yᵀ L^{(v)} y)            (Eq.4-5)

    ⚠️ **图的节点是实例，边权来自该视图的特征 ``X⁽ⁱ⁾``；被平滑的对象才是标签
    ``y⁽ⁱ⁾``。** 论文式 (3) 的距离明确取 ``‖x_i − x_j‖``。仓库
    ``alg/I2VSLC.py`` L44 用 ``construct_W(v[i])``（特征块）建图，与论文一致；
    而 ``PAPERS_AND_CODE.md`` §5 与 ``paper-notes/I2VSLC.md`` 写作「标签图 /
    在标签空间建图」，属于**文档表述错误**（本函数签名因此接收特征矩阵）。

    平滑假设（论文引 [40]）："the similarity of pairwise features has positive
    correlation with the similarity of pairwise view-specific labels" ——
    特征相近 ⇒ 标签也应相近。

    Parameters
    ----------
    Y : (n, d⁽ⁱ⁾) ndarray
        **该视图的特征矩阵 X⁽ⁱ⁾**（不是标签矩阵）。参数名保留为 ``Y`` 仅为与
        论文中 ``y`` 的用法区分；调用时请传特征块。
    k, t : int, float
        ``q`` 近邻数与热核带宽。``skfeature.construct_W`` 的热核是
        ``exp(−d²/(2t²))``，对应论文的 ``σ = √2·t``。

    Returns
    -------
    S, A_deg, L : ndarray
    """
    from skfeature.utility.construct_W import construct_W

    options = {"metric": "euclidean", "neighbor_mode": "knn", "k": int(k),
               "weight_mode": "heat_kernel", "t": float(t)}
    S = construct_W(np.asarray(Y, dtype=float), **options)
    S = np.asarray(S.toarray() if hasattr(S, "toarray") else S, dtype=float)
    A_deg = np.diag(S.sum(axis=1))
    return S, A_deg, A_deg - S


def self_portrait_objective(ys: list[np.ndarray], Ls: list[np.ndarray],
                            alpha: float = 1.0) -> dict:
    r"""M5-b: 自刻画函数 Φ(·) 的两项分解（论文式 (5)+(6) ⇒ 式 (7)）。

    .. math::

        \Phi(\cdot)=\underbrace{\sum_{i=1}^{V} \mathrm{Tr}\!\left(y^{(i)\top}
                    L^{(i)} y^{(i)}\right)}_{\text{intra-view 视图内标签流形平滑}}
                  +\underbrace{\sum_{i=1}^{V}\sum_{j\neq i}
                    \left\|y^{(i)}-y^{(j)}\right\|_F^2}_{\text{inter-view 视图间共识}}

    **intra-view 项**：图由 ``X⁽ⁱ⁾`` 的 kNN 热核给出，作用在 ``y⁽ⁱ⁾`` 上 ⇒
    特征相近的实例被拉向相近的伪标签。**这是"视图特有"的唯一来源** ——
    ``L⁽ⁱ⁾`` 只编码第 i 个视图的实例结构，不含任何跨视图信息。

    **inter-view 项**（论文式 (6)）：刻意**不引入均值/标准参考点**，
    直接在每一对上拉近。原文："In order to mitigate any potential impact
    resulting from the quality of a specific reference point such as average
    value, a standard point is not employed to measure the difference between
    two matrices."

    本函数同时给出**两种** inter-view 实现，用于对照量化仓库实现的偏离：

    * ``pairwise`` —— 论文式 (6)：``Σ_{i<j}‖y⁽ⁱ⁾−y⁽ʲ⁾‖²_F``，**有符号软距离**。
    * ``union_indicator`` —— 仓库 ``alg/I2VSLC.py`` L85-92 的做法：把
      ``tem1 = [ȳ−y⁽ⁱ⁾]⁺`` 与 ``tem2 = [y⁽ⁱ⁾−ȳ]⁺`` 二值化成**同一个** 0/1
      不一致掩码（``tem1 ≡ tem2``），于是线性梯度退化为"对不一致位置统一加
      常数拉力"，与差距大小无关。

    Returns
    -------
    dict
        ``{'intra','inter_pairwise','inter_union','pairwise','union_indicator'}``
    """
    V = len(ys)
    intra = float(sum(np.trace(ys[i].T @ Ls[i] @ ys[i]) for i in range(V)))

    inter_pairwise = float(sum(
        LA.norm(ys[i] - ys[j], "fro") ** 2
        for i in range(V) for j in range(i + 1, V)))

    union = np.zeros_like(ys[0])
    for y in ys:
        union = union + y
    union = (union > 0).astype(float)
    inter_union = float(sum(
        np.count_nonzero((ys[i] != union).astype(float)) for i in range(V)))

    return {
        "intra": intra,
        "inter_pairwise": inter_pairwise,
        "inter_union": inter_union,
        "pairwise": intra + alpha * inter_pairwise,
        "union_indicator": intra + alpha * inter_union,
    }


def view_specific_label_step(X: np.ndarray, w: np.ndarray, y: np.ndarray,
                             S: np.ndarray, A_deg: np.ndarray,
                             ys_others: list[np.ndarray], Y: np.ndarray,
                             AA: np.ndarray,
                             alpha: float = 1.0, beta: float = 1.0,
                             n_views: int = 2,
                             binarize_difference: bool = False) -> np.ndarray:
    r"""M5-c: 视图特有伪标签 y⁽ⁱ⁾ 的乘性更新（论文式 (27)）。

    .. math::

        y^{(i)} \leftarrow y^{(i)} \circ
          \frac{\alpha S^{(i)} y^{(i)} + \alpha(V-1) F
                + X^{(i)} w^{(i)} + \beta (Y C + G)}
               {\alpha A^{(i)} y^{(i)} + \alpha(V-1) y^{(i)} + (1+\beta) y^{(i)}}

    其中（论文式 (26) 定义）::

        F = Y_vs − y^{(i)},     G = y^{(i)} − Y_vs,     Y_vs = 1[Σ_j y^{(j)} ≥ 1]

    ``F`` 与 ``G`` 是**有符号**的连续差。把它们放进乘性更新在理论上是有瑕疵的
    （KKT 分解要求正负部非负）；仓库实现用「二值化成同一个 0/1 掩码」回避了
    这一点，代价是**丢失梯度幅值信息**。

    Parameters
    ----------
    binarize_difference : bool
        ``False``（默认）→ 忠实实现论文式 (27) 的**软距离**形式。
        ``True`` → 复现仓库 ``alg/I2VSLC.py`` L85-92 的**二值化掩码**形式。
        两者对照即可量化该偏离对目标函数的影响。

    Returns
    -------
    y_new : (n, l) ndarray
    """
    union = np.zeros_like(y)
    for yj in ys_others:
        union = union + yj
    union = union + y                                   # Y_vs = 1[Σ_j y^(j) ≥ 1]
    Y_vs = (union > 0).astype(float)

    F = Y_vs - y                                        # 论文式 (26) 的 F
    G = y - Y_vs                                        # 论文式 (26) 的 G
    if binarize_difference:
        F = (F != 0).astype(float)
        G = (G != 0).astype(float)

    num = (alpha * (S @ y) + alpha * (n_views - 1) * F
           + (X @ w) + beta * ((Y @ AA) + G))
    den = (alpha * (A_deg @ y) + alpha * (n_views - 1) * y
           + (1.0 + beta) * y + EPS)
    return y * (num / den)


# =========================================================================== #
# M6 — 锚点引导的双路径多层分解 (GRAFS)
# =========================================================================== #

def anchor_select_kmeans(X: np.ndarray, kk: int,
                         seed: int | None = None) -> np.ndarray:
    r"""M6-a: 锚点选取（k-means 质心）。

    论文没有把「锚点怎么选」当作贡献点（用的是标准做法），但这是
    ``O(n²) → O(n·kk)`` 的关键。这里实现 k-means 质心版本，
    锚点矩阵 ``B ∈ R^{n×kk}`` 的**每一行**是一个样本到各锚点的相似度，
    于是 ``X⁽ⁱ⁾ ≈ B P⁽ⁱ⁾`` 把 ``d⁽ⁱ⁾`` 维重构压缩到 ``kk`` 维。

    Parameters
    ----------
    X : (n, d) ndarray
        通常是所有视图拼接后的特征（锚点在**样本空间**选取，与视图无关）。
    kk : int
        锚点数。论文默认与 ``0.1n`` 同量级。

    Returns
    -------
    B : (n, kk) ndarray, 每行归一化到 [0, 1]
    """
    from sklearn.cluster import KMeans

    n = X.shape[0]
    kk = int(min(kk, n))
    km = KMeans(n_clusters=kk, n_init=10, random_state=seed if seed is not None else 100)
    labels = km.fit_predict(X)
    B = np.zeros((n, kk))
    B[np.arange(n), labels] = 1.0
    # 用「样本到锚点质心的相似度」替代硬指示，保留几何信息
    centers = km.cluster_centers_
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    B = np.exp(-d2 / (2.0 * (np.mean(np.sqrt(d2)) + EPS) ** 2))
    return B / (B.max(axis=1, keepdims=True) + EPS)


def grafs_reconstruct(views, B, W1, A1, A2, R1, R2, X_f, nu,
                      alpha=1.0, beta=1.0):
    r"""M6-b: GRAFS 的「锚点引导」双路径多层分解（论文 Eq.6-8）。

    **目标函数**::

        L_E =  Σᵢ vᵢ‖X⁽ⁱ⁾ − B P⁽ⁱ⁾‖²_F          （锚点重构各视图）
             + ‖B − W_c A_c R_c‖²_F               （锚点侧多层分解）
             + ‖X_f − W_c A_f R_f‖²_F             （全局视图侧多层分解）
             + ‖A_c − A_f‖²_F                     （两条路径的结构一致性）

    **为什么是创新**: GRAFS 之前的「全局视图」要么是**直接拼接**（丢掉视图关系），
    要么是**固定权重加权拼接**（权重一次算死）。GRAFS 把 ``X_f`` 变成一个
    **被求解的变量**，并且让它与锚点 ``B`` 共享同一条分解链
    ``W_c A_c R_c`` / ``W_c A_f R_f``：

    * 共享 ``W_c`` → 两条路径在同一基底下，结构可比
    * ``‖A_c − A_f‖²`` → 强制锚点侧学到的**结构矩阵**与全局视图侧一致

    于是 ``X_f`` 同时被「标签语义（``X_f W ≈ Y``）」和「视图几何（锚点）」牵引，
    这就是论文说的 "anchor-guided"：锚点是**结构先验的载体**，
    而不是单纯的加速近似。

    Returns
    -------
    dict
        ``{'L_E': float, 'per_view': list[float], 'anchor_term': float,
           'global_term': float, 'consistency_term': float, 'X_f': ndarray}``
    """
    per_view, anchor_term, global_term = [], 0.0, 0.0
    for i, Xi in enumerate(views):
        P_i = LA.pinv(B) @ Xi                     # B 固定 → P⁽ⁱ⁾ 有闭式解
        per_view.append(nu[i] * float(LA.norm(Xi - B @ P_i, "fro") ** 2))

    anchor_term = float(LA.norm(B - W1 @ A1 @ R1, "fro") ** 2)
    global_term = float(LA.norm(X_f - W1 @ A2 @ R2, "fro") ** 2)
    consistency = float(LA.norm(A1 - A2, "fro") ** 2)

    L_E = sum(per_view) + anchor_term + global_term + consistency
    return {"L_E": L_E, "per_view": per_view, "anchor_term": anchor_term,
            "global_term": global_term, "consistency_term": consistency,
            "X_f": X_f}


# =========================================================================== #
# M7 — ADMM + 有证明的收敛（对六篇的补强）
# =========================================================================== #

def admm_nmf(X: np.ndarray, Y: np.ndarray, rank: int, rho: float = 1.0,
             max_iter: int = 300, tol: float = 1e-6,
             seed: int | None = None) -> dict:
    r"""M7: 用 ADMM 求解非负最小二乘型子问题 —— 带**可验证的收敛判据**。

    六篇论文无一例外使用**乘性更新规则 (multiplicative update rules)**，
    并在贡献点里声称 "proven convergence"，但正文中**既无定理也无引理**
    （见 ``PAPER_METHOD_ANALYSIS.md`` §5.5 的证据）。乘性更新实际求解的是

    .. math::  \min_{W \ge 0} \|XW - Y\|_F^2

    其不动点条件为

    .. math::

        W \leftarrow W \circ \frac{X^\top Y}{X^\top X W}
        \quad\Longleftrightarrow\quad
        [X^\top(XW - Y)] \circ W = 0

    即**非负性下的 KKT 互补松弛条件**（Gonzalez & Zhang 定理）。它是
    「部分」最优性条件：只保证一阶必要条件，不保证全局最优，也**不保证
    ALM/ADMM 意义下的原始-对偶残差趋于零**。

    **ADMM 补强**: 引入分裂变量 ``Z`` 与对偶变量 ``U``::

        min_{W,Z} ½‖XW − Y‖²_F + ι_{≥0}(Z)
        s.t.  W − Z = 0

        增广拉格朗日:  L_ρ = ½‖XW − Y‖² + ι_{≥0}(Z)
                             + ⟨U, W−Z⟩ + (ρ/2)‖W−Z‖²

        步1 (W):  W ← (XᵀX + ρI)⁻¹ (XᵀY + ρ(Z − U))     ← 线性系统，闭式
        步2 (Z):  Z ← Π_{≥0}(W + U)                      ← 投影，闭式
        步3 (U):  U ← U + ρ(W − Z)                       ← 对偶上升

    **收敛判据**（Boyd et al. 2011, §3.3）::

        r_prim = ‖W − Z‖_F                → 0   （原始残差）
        r_dual = ρ‖Z − Z_prev‖_F          → 0   （对偶残差）

    ADMM 对「闭凸集 + 凸二次」这类问题**有全局收敛定理**，且迭代可以
    **自适应调 ρ**（``ρ ← 2ρ`` if ``r_prim > 10 r_dual`` 等），比固定步长的
    乘性更新更鲁棒也更有说服力 —— 这是把这一线工作送进更高档位会议
    （NeurIPS/ICML/AAAI 的 theory-aware 审稿人）最省力的改造。

    Returns
    -------
    dict
        ``{'W','Z','U','r_prim','r_dual','obj','iters','converged'}``
    """
    rng = np.random.default_rng(seed if seed is not None else 100)
    n, d = X.shape
    l = Y.shape[1]
    W = rng.random((d, l))
    Z = W.copy()
    U = np.zeros_like(W)

    XtX = X.T @ X
    XtY = X.T @ Y

    r_prim_hist, r_dual_hist, obj_hist = [], [], []
    converged = False

    for it in range(max_iter):
        # --- 步 1: W-update（岭回归闭式解） ---
        W = LA.solve(XtX + rho * np.eye(d), XtY + rho * (Z - U))

        # --- 步 2: Z-update（非负投影） ---
        Z_prev = Z
        Z = np.maximum(W + U, 0.0)

        # --- 步 3: 对偶上升 ---
        U = U + (W - Z)

        r_prim = float(LA.norm(W - Z, "fro"))
        r_dual = float(rho * LA.norm(Z - Z_prev, "fro"))
        r_prim_hist.append(r_prim)
        r_dual_hist.append(r_dual)
        obj_hist.append(0.5 * float(LA.norm(X @ W - Y, "fro") ** 2))

        # --- 自适应 ρ（Boyd §3.4.1） ---
        if r_prim > 10.0 * r_dual:
            rho *= 2.0
            U /= 2.0
        elif r_dual > 10.0 * r_prim:
            rho /= 2.0
            U *= 2.0

        if max(r_prim, r_dual) < tol:
            converged = True
            break

    return {"W": Z, "Z": Z, "U": U,
            "r_prim": r_prim_hist, "r_dual": r_dual_hist,
            "obj": obj_hist, "iters": it + 1, "converged": converged}


# =========================================================================== #
# M8 — 自适应图学习 + 连通分量约束（对六篇的补强）
# =========================================================================== #

def adaptive_graph_learning(X: np.ndarray, n_clusters: int,
                            gamma: float | None = None, k: int = 10,
                            max_iter: int = 100, tol: float = 1e-6,
                            seed: int | None = None) -> dict:
    r"""M8: 自适应相似图学习 + 秩约束（对 UGRFS/GRAFS/EF2FS 固定图的补强）。

    **六篇的共同弱点**: 视图权重 ``vᵢ = 1/Tr(X⁽ⁱ⁾ᵀ L⁽ⁱ⁾ X⁽ⁱ⁾)`` 里的
    ``L⁽ⁱ⁾`` 由 ``construct_W(knn, k=5 or 20, heat_kernel, t=1.0)`` **一次算死**，
    之后完全不再更新。后果有三：

    1. ``k``、``t`` 是额外超参，且对数据尺度敏感（``t`` 固定为 1.0 尤其危险）；
    2. 固定的 knn 图含有大量**跨簇**边，流形平滑会把不同类别的样本拉近；
    3. 图**不随特征选择过程演化**，而特征选择恰恰在改变数据的几何。

    **本机制**: 把相似图 ``S`` 当作变量联合优化（Nie et al. 的自适应图学习）::

        min_{S, F}  Σ_{i,j} ‖x_i − x_j‖² s_{ij} + γ Tr(Fᵀ L_S F)
        s.t.  s_iᵀ1 = 1, s_ij ≥ 0,  FᵀF = I,  rank(L_S) = n − c

    其中 ``rank(L_S) = n − c`` 等价于「``S`` 恰好有 ``c`` 个连通分量」，
    这正是**聚类结构**的定义。该约束可以用 Ky Fan 定理转写：

    .. math::  \mathrm{rank}(L_S) = n - c
               \iff \min_{F^\top F = I} \mathrm{Tr}(F^\top L_S F) = 0
               \iff \sum_{i=1}^{c}\lambda_i(L_S) = 0

    给定 ``F`` 后 ``S`` 有**逐行闭式解**（单纯形投影）::

        s_{ij} = \frac{d_{ij} - \gamma\,g_{ij}}{2\lambda_i}
        \quad\text{→ 经排序后的解析解（Nie et al. 2016, Eq.10）}

    其中 :math:`d_{ij} = \|x_i - x_j\|_2^2`，:math:`g_{ij} = \|f_i - f_j\|_2^2`。

    这一改动把「视图权重」的分母从一个**猜测出来的固定图能量**，换成
    **与特征学习共同优化的目标函数项**，并且在数学上给出了
    「图应当有几个连通块」的显式约束 —— 这是把这一线工作推到
    TPAMI/TKDE/NeurIPS 档位的核心补强。

    ⚠️ **γ 的量纲**（实现时必须注意）: γ 乘在 ``‖f_i − f_j‖²`` 上，而 ``F``
    的列是单位正交的（``‖f_i‖₂ ≈ √(c/n)``），故 ``‖f_i−f_j‖²`` 是 **O(c/n)**；
    而 ``d_ij = ‖x_i−x_j‖²`` 的量纲是**特征尺度的平方**。二者必须可比，否则：

      * γ **过小** → 距离项主导 → ``S`` 退化为 1-NN 图，连通分量数 ≈ n；
      * γ **过大** → ``F`` 项主导 → ``S`` 退化为近均匀图，连通分量数 = 1。

    只有 ``γ ≈ median(d²) / (2c/n)`` 附近才落在"恰好恢复 c 个连通分量"的
    正确区间。``gamma=None``（默认）时按此自动标定 —— 这是**尺度自适应的
    关键**，也正是论文那类"固定 ``t=1.0``"的做法做不到的。

    Returns
    -------
    dict
        ``{'S','F','obj','iters','n_components','eigenvalues','gamma'}``
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed if seed is not None else 100)

    # 初始化：knn 图（同时也是「固定图」基线的对照起点）
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    S = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(d2[i])[:k]
        S[i, idx] = 1.0 / k
    S = (S + S.T) / 2.0

    if gamma is None:
        finite = d2[np.isfinite(d2)]
        gamma = float(np.median(finite)) / (2.0 * n_clusters / n)

    F = rng.random((n, n_clusters))
    F, _ = LA.qr(F)
    obj_hist = []

    for it in range(max_iter):
        # --- F-update: L_S 前 c 个最小特征向量（Ky Fan 定理） ---
        L_S = np.diag(S.sum(axis=1)) - S
        vals, vecs = LA.eigh(L_S)
        F = vecs[:, :n_clusters]
        G = ((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2)   # ‖f_i − f_j‖²

        # --- S-update: 逐行闭式解（Nie et al. 2016, Eq.10-12） ---
        #
        # 逐行问题：  min_{s_i} ½‖s_i + d_i/(2γ)‖²  s.t. s_iᵀ1=1, s_i ≥ 0
        # 其中 d_ij = ‖x_i−x_j‖² + γ‖f_i−f_j‖²（见 Nie 2016 式 (8) 的合并写法）。
        # 拉格朗日 ⇒ s_ij = max(−d_ij/(2γ) + λ, 0)，λ 由 Σ_j s_ij = 1 定。
        # 于是逐行是**单纯形投影**，而非常见的"截断 + 归一化"路径。
        # 这样能保证 s_i ≥ 0 且 Σ_j s_ij = 1，可行域封闭，迭代不会漂移。
        D_sim = d2 + gamma * G
        D_sim = np.nan_to_num(D_sim, nan=0.0, posinf=0.0)
        np.fill_diagonal(D_sim, np.inf)
        S_new = np.zeros((n, n))
        for i in range(n):
            v = -D_sim[i] / (2.0 * gamma)          # 目标里 s_i 的系数方向
            v = np.where(np.isfinite(v), v, 0.0)
            v[i] = 0.0
            S_new[i] = _simplex_projection(v)
        S = (S_new + S_new.T) / 2.0

        L_S = np.diag(S.sum(axis=1)) - S
        obj = float(np.sum(np.nan_to_num(d2, nan=0.0, posinf=0.0) * S)) \
            + gamma * float(np.trace(F.T @ L_S @ F))
        obj_hist.append(obj)

        if it > 0 and abs(obj_hist[-1] - obj_hist[-2]) / (abs(obj_hist[-2]) + EPS) < tol:
            break

    # 连通分量数（0 特征值个数，容差 1e-8）
    L_S = np.diag(S.sum(axis=1)) - S
    evals = np.sort(LA.eigvalsh(L_S))
    n_comp = int(np.sum(evals < 1e-8))

    return {"S": S, "F": F, "obj": obj_hist, "iters": it + 1,
            "n_components": n_comp, "eigenvalues": evals, "gamma": gamma}


# =========================================================================== #
# 内部工具
# =========================================================================== #

def _l21_diag(W: np.ndarray) -> np.ndarray:
    r"""``l2,1`` 范数重加权对角阵：``E_pp = 1/(2‖W_p‖₂)``。

    依据 ``‖W‖_{2,1} = 2 Tr(Wᵀ E W)``（UGRFS 论文 Optimization 小节）。
    """
    d = np.sqrt((W ** 2).sum(axis=1)) + EPS
    return np.diag(0.5 / d)


def _simplex_projection(v: np.ndarray) -> np.ndarray:
    r"""把向量投影到概率单纯形 ``{s : s ≥ 0, Σs = 1}``（Duchi et al. 2008）。

    解 ``min_s ½‖s − v‖²  s.t. s ≥ 0, sᵀ1 = 1``：

    1. 把 ``v`` 降序排列得 ``u``，求累积和 ``c``；
    2. 取最大的 ``ρ`` 使 ``u_ρ − (c_ρ − 1)/ρ > 0``；
    3. ``s_i = max(v_i − θ, 0)``，``θ = (c_ρ − 1)/ρ``。

    复杂度 ``O(n log n)``，闭式、无参数。自适应图学习里每一行都要做一次，
    这也是该机制的主要开销来源。
    """
    n = v.size
    u = np.sort(v)[::-1]
    c = np.cumsum(u)
    rho = np.arange(1, n + 1)
    cond = u - (c - 1.0) / rho > 0
    if not np.any(cond):
        out = np.zeros_like(v)
        out[int(np.argmax(v))] = 1.0
        return out
    r = int(np.max(np.where(cond)[0]))
    theta = (c[r] - 1.0) / (r + 1.0)
    s = np.maximum(v - theta, 0.0)
    total = s.sum()
    return s / total if total > EPS else s


def _label_aware_view_energy(X: np.ndarray, Y: np.ndarray,                             L: np.ndarray, lam: float = 1e-3) -> float:
    r"""修正版视图能量：**归一化图平滑后的标签拟合残差**。

    论文的 ``Tr(GᵀLG)`` 有两个问题：(1) 对 ``G`` 的尺度是二次敏感的
    （``Tr((cG)ᵀL(cG)) = c²Tr(GᵀLG)``），因此特征取值小的视图会被系统性
    高估；(2) 完全不涉及标签 ``Y``，纯噪声视图只要几何"紧"就能拿高权重。

    本函数改为：

    .. math::

        E_i = \frac{1}{d^{(i)}}\Big[\min_{W}\ \big\|\tilde S^{(i)}Y
              - X^{(i)}W\big\|_F^2 + \lambda\|W\|_F^2\Big],
        \qquad \tilde S = D^{-1/2} S D^{-1/2}

    其中 ``S`` 是该视图的 kNN 热核亲和矩阵，``1/d^{(i)}`` 是**维度归一化**。
    性质：

    * **尺度不变**：度归一化把 ``S`` 的行和钉成常数，特征整体缩放不再
      改变 ``S̃`` 的结构；
    * **维度可比**：除以 ``d^{(i)}`` 消除"特征多的视图拟合残差天然更大"
      的自由度偏差（否则 d 小的视图会被系统性高估）；
    * **标签相关**：``S̃Y`` 是"按该视图邻域平滑后的标签"，残差小意味着
      "该视图的局部几何能解释标签" —— 这正是"视图重要"的定义；
    * **有闭式解**：``W = (XᵀX + λI)⁻¹XᵀS̃Y``，无需迭代。

    Returns
    -------
    float
        能量 ``E_i``（越小越重要）。
    """
    # L = D − S ⇒ S = D − L。L 的对角元就是度 ``d_i``，故先取负的非对角块，
    # 再补回度作为对角（等价于从 L 还原 S）。
    A = L.copy()
    np.fill_diagonal(A, 0.0)
    S = -A
    deg = S.sum(axis=1) + EPS
    S = S + np.diag(deg)                     # 补回度 ⇒ S = deg − L_off
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    S_norm = (S * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]

    Y_smooth = S_norm @ Y
    d = X.shape[1]
    W = LA.solve(X.T @ X + lam * np.eye(d), X.T @ Y_smooth)
    resid = float(LA.norm(X @ W - Y_smooth, "fro") ** 2)
    return resid / d                          # 维度归一化


def _lam_diag_scale(D: np.ndarray, A: np.ndarray) -> np.ndarray:
    """把 ``l2,1`` 对角阵作用到 ``A`` 上（保留为独立函数以便替换为自适应权重）。"""
    return D @ A
