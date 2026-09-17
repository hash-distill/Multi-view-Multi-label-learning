import time

import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
from skfeature.utility.construct_W import construct_W

from alg._util import as_int_indices, dense, make_rng, max_iter, view_dim

"""
Embedded Feature Fusion for multi-view multi-label feature selection
(EF2FS, Pattern Recognition 157 (2025) 110888).

Fix over the original implementation: the view-weight step referenced
``Lx_lst`` before it existed, raising NameError on the very first iteration.
The graph Laplacians are now built once, before the optimisation loop, with the
same construction used by I2VSLC (k=5, heat kernel) so the two algorithms remain
comparable.
"""

eps = 2.2204e-16


def EF2FS(X, x_view, Y, dataset, alpha, beta, gamma, lamb, V_dim, seed=None):
    time_start = time.time()
    n_view = len(x_view)
    num, dim = X.shape
    num, label_num = Y.shape
    w = []  # per-view projection matrices
    m = []
    v = []

    rng = make_rng(seed)
    for i in range(n_view):
        m.append(view_dim(x_view, i))
        w.append(rng.random((m[i], V_dim)))

    t1 = 0
    for i in range(n_view):
        v.append(X[:, t1:(t1 + m[i])])
        t1 = t1 + m[i]

    # graph Laplacian of every view (this is what the weighting step needs)
    Lx_lst = []
    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5,
               'weight_mode': 'heat_kernel', 't': 1.0}
    for i in range(n_view):
        Sx = dense(construct_W(v[i], **options))
        Ax = np.diag(np.sum(Sx, 0))
        Lx_lst.append(Ax - Sx)

    V = rng.random((num, V_dim))
    B = rng.random((label_num, V_dim))
    A = rng.random((dim, V_dim))

    obj = []
    obji = 1
    iter = 0
    MAX_ITER = max_iter(100)

    while 1:
        # 1. view weight coefficients
        # A view whose Laplacian energy is zero (degenerate graph) would make
        # 1/energy explode; such a view gets weight 0 and the remaining views
        # share the budget. The original code divided unguarded here.
        nu_temp = np.empty(n_view)
        nu_all = 0.0
        for i in range(n_view):
            energy = float(np.trace(V.T @ Lx_lst[i] @ V))
            nu_temp[i] = 1.0 / energy if energy > eps else 0.0
            nu_all = nu_all + nu_temp[i]
        if nu_all > eps:
            nu = nu_temp / nu_all
        else:
            nu = np.full(n_view, 1.0 / n_view)

        new_X = np.concatenate([v[i] * nu[i] for i in range(n_view)], axis=1)

        # 2. update B, V, A, w
        B = np.multiply(B, np.true_divide(np.dot(Y.T, V), np.dot(np.dot(B, V.T), V) + eps))

        xw = np.dot(v[0], w[0])
        for i in range(1, n_view):
            xw = xw + np.dot(v[i], w[i])

        V = np.multiply(V, np.true_divide(
            alpha * np.dot(Y, B) + beta * xw + np.dot(new_X, A),
            alpha * np.dot(np.dot(V, B.T), B) + (n_view * beta + 1) * V + eps))

        Atmp = np.sqrt(np.sum(np.multiply(A, A), 1) + eps)
        D = np.diag((0.5 / Atmp).flat)
        A = np.multiply(A, np.true_divide(
            np.dot(X.T, V) + gamma * np.concatenate(w, axis=0),
            np.dot(np.dot(X.T, X), A) + gamma * A + lamb * np.dot(D, A) + eps))

        t1 = 0
        for i in range(n_view):
            new_a = A[t1:(t1 + m[i]), :]
            w[i] = np.multiply(w[i], np.true_divide(
                beta * np.dot(v[i].T, V) + gamma * new_a,
                beta * np.dot(np.dot(v[i].T, v[i]), w[i]) + gamma * w[i] + eps))
            t1 = t1 + m[i]

        # 3. objective
        temp1 = pow(LA.norm(np.dot(new_X, A) - V, 'fro'), 2)
        temp2 = pow(LA.norm(Y - np.dot(V, B.T), 'fro'), 2)

        temp3 = 0
        for i in range(n_view):
            temp3 = temp3 + pow(LA.norm(np.dot(v[i], w[i]) - V, 'fro'), 2)

        temp4 = 0
        t1 = 0
        for i in range(n_view):
            temp4 = temp4 + pow(LA.norm(A[t1:(t1 + m[i]), :] - w[i], 'fro'), 2)
            t1 = t1 + m[i]

        temp5 = np.sum(np.sqrt(np.sum(A * A, 1)))

        objectives = temp1 + alpha * temp2 + beta * temp3 + gamma * temp4 + lamb * temp5
        if iter == 0:
            obji = objectives / 2
        obj.append(objectives)

        cver = abs((objectives - obji) / float(obji))
        obji = objectives

        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)):
            break

    time_end = time.time()
    running_time = time_end - time_start

    w_2 = LA.norm(A, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    param = dict(alpha=alpha, beta=beta, gamma=gamma, lamb=lamb, V_dim=V_dim)

    record = dict()
    record['method'] = 'EF2FS'
    record['dataset'] = dataset
    record['param'] = param
    record['running_time'] = running_time
    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = as_int_indices(f_idx).tolist()
    record['selected_num'] = None

    return record, iter
