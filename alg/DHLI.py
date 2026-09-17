import time
import numpy as np

from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler

from alg._util import as_int_indices, make_rng, max_iter, view_dim


eps = 2.2204e-16


def DHLI(X, x_view, Y, dataset, alpha, beta, gamma, lamb, seed=None):
    time_start = time.time()
    n_view = len(x_view)
    num, label_num = Y.shape
    w = []
    m = []
    y = []
    u = []

    rng = make_rng(seed)
    for i in range(n_view):
        m.append(view_dim(x_view, i))
        w.append(rng.random((m[i], label_num)))
        u.append(rng.random((m[i], label_num)))
        y.append(rng.integers(2, size=(num, label_num)))

    sum_y = 0
    for i in range(n_view):
        sum_y = sum_y + y[i]

    sum_yt = sum_y.copy()
    sum_yt[sum_yt == 0] = 0
    sum_yt[sum_yt > 0] = 1

    # 切分：
    x = []
    for i in range(0, len(m)):
        if i == 0:
            x.append(X[:, :m[i]])
        else:
            x.append(X[:, m[i - 1]:(m[i - 1] + m[i])])

    Y_c = rng.integers(2, size=(num, label_num)).astype(float)
    Y_n = rng.integers(2, size=(num, label_num)).astype(float)

    cver_lst = []

    obj = []
    obji = 1
    iter = 0
    MAX_ITER = max_iter(500)

    Y_spe = Y - Y_c - Y_n
    Y_spe[Y_spe <= 0] = 0
    Y_spe[Y_spe > 0] = 1

    while 1:
        t1 = 0

        for i in range(n_view):
            d_temp = np.sqrt(np.sum(np.multiply(w[i], w[i]), 1) + eps)
            d1 = 0.5 / d_temp
            D = np.diag(d1.flat)

            e_temp = np.sqrt(np.sum(np.multiply(u[i], u[i]), 1) + eps)
            e1 = 0.5 / e_temp
            E = np.diag(e1.flat)

            w[i] = np.multiply(w[i], np.true_divide(np.dot(x[i].T, Y_c), np.dot
            (x[i].T, np.dot(x[i], w[i])) + gamma*np.dot(D, w[i]) + eps))
            u[i] = np.multiply(u[i], np.true_divide(np.dot(x[i].T, y[i]), np.dot
            (x[i].T, np.dot(x[i], u[i])) + gamma * np.dot(E, u[i]) + eps))

            y_o = sum_yt - y[i]
            y_o[y_o == 0] = 0# 按sum_yt来说不存在负值
            y_o[y_o > 0] = 1

            # B is the Hadamard product of all *other* views' label matrices.
            # The original code initialised B = 0, so `B = B * y[i]` stayed 0
            # forever and the `lamb * y[i] * B * B` term below always vanished.
            # The multiplicative identity for the product is 1.
            B = 1
            for j in range(n_view):
                if i != j:
                    B = B * y[j]

            tem1 = Y_spe - y_o
            tem1[tem1 == 0] = 0
            tem1[tem1 > 0] = 1
            tem1[tem1 < 0] = 1
            y[i] = np.multiply(y[i], np.true_divide(np.dot(x[i],u[i]) + lamb * tem1, y[i] + lamb * y[i] + lamb * y[i] * B * B+eps))

            t1 = t1 + np.dot(x[i], w[i])


        for i in range(n_view):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(y[i])
            y[i] = scaler.transform(y[i])
            y[i][y[i] <= 0.5] = 0
            y[i][y[i] > 0.5] = 1

        sum_y = 0
        for i in range(n_view):
            sum_y = sum_y + y[i]

        sum_yt = sum_y.copy()
        sum_yt[sum_yt == 0] = 0
        sum_yt[sum_yt > 0] = 1


        tem2 = Y-Y_n-sum_yt
        tem2[tem2 == 0] = 0
        tem2[tem2 > 0] = 1
        tem2[tem2 < 0] = 1
        Y_c = np.multiply(Y_c, np.true_divide(t1 + alpha* Y + lamb*(tem2), n_view*Y_c + alpha * Y_c + lamb * Y_c +eps))

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(Y_c)
        Y_c = scaler.transform(Y_c)
        Y_c[Y_c <= 0.5] = 0
        Y_c[Y_c > 0.5] = 1


        tem3 = Y - Y_c - sum_yt
        tem3[tem3 == 0] = 0
        tem3[tem3 > 0] = 1
        tem3[tem3 < 0] = 1
        Q = np.true_divide(1, 2 * abs(Y_n) + eps)
        Y_n = np.multiply(Y_n, np.true_divide(lamb * (tem3), beta * Q * Y_n + lamb * Y_n + eps))



        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(Y_n)
        Y_n = scaler.transform(Y_n)
        Y_n[Y_n<= 0.5] = 0
        Y_n[Y_n > 0.5] = 1


        # objectives:
        temp1 = 0
        temp2 = 0

        for i in range(n_view):
            temp1 = temp1 + pow(LA.norm(np.dot(x[i], w[i]) - Y_c, 'fro'), 2)
        for i in range(n_view):
            temp2 = temp2 + pow(LA.norm(np.dot(x[i], u[i]) - y[i], 'fro'), 2)


        temp3 = alpha * pow(LA.norm(Y-Y_c, 'fro'),2)
        temp4 = beta * 2 *np.sum(abs(Y_n))

        W = np.concatenate(w, axis=0)
        d_temp = np.sqrt(np.sum(np.multiply(W, W), 1) + eps)
        d1 = 0.5 / d_temp
        D_all = np.diag(d1.flat)
        temp5 = gamma * 2 * np.trace(np.dot(np.dot(W.T, D_all), W))

        U = np.concatenate(u, axis=0)
        e_temp = np.sqrt(np.sum(np.multiply(U, U), 1) + eps)
        e1 = 0.5 / e_temp
        E_all = np.diag(e1.flat)
        temp6 = gamma * 2 * np.trace(np.dot(np.dot(U.T, E_all), U))

        tem7 = y[0]
        for i in range(1,n_view):
            tem7 = tem7 * y[i]
        temp7 = lamb * pow(LA.norm(tem7,'fro'),2)

        Y_spe = Y - Y_c - Y_n
        Y_spe[Y_spe == 0] = 0
        Y_spe[Y_spe > 0] = 1
        Y_spe[Y_spe < 0] = 1

        tem8 = sum_yt - Y_spe
        tem8[tem8 == 0] = 0
        tem8[tem8 > 0] = 1
        tem8[tem8 < 0] = 1
        temp8 = lamb * pow(LA.norm(tem8, 'fro'),2)

        objectives = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8
        if iter == 0:
            obji = objectives /2
        obj.append(objectives)

        cver = abs((objectives - obji) / float(obji))
        obji = objectives
        cver_lst.append(cver)


        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == MAX_ITER)):
            break

    time_end = time.time()
    running_time = time_end - time_start


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(W)
    tW = scaler.transform(W)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(U)
    tU = scaler.transform(U)

    A = tW + tU
    w_2 = LA.norm(A, ord=2, axis=1)
    f_idx = np.argsort(-w_2)

    # 记录参数
    param = dict()
    param['alpha'] = alpha
    param['beta'] = beta
    param['gamma'] = gamma
    param['lamb'] = lamb

    record = dict()
    record['method'] = 'DHLI'
    record['dataset'] = dataset
    record['param'] = param
    record['running_time'] = running_time
    record['obj_value'] = np.array(obj).reshape(1, len(obj))
    record['idx'] = as_int_indices(f_idx).tolist()
    record['selected_num'] = None

    return record, iter
