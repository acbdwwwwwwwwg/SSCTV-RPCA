# lrasr.py
# -*- coding: utf-8 -*-
import numpy as np

def _svd_threshold(X, r):
    """核范数近端：soft-threshold on singular values."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_shrink = np.maximum(s - r, 0.0)
    # 仅保留正奇异值对应的分量
    nz = np.count_nonzero(s_shrink > 0)
    if nz == 0:
        return np.zeros_like(X)
    return (U[:, :nz] * s_shrink[:nz]) @ Vt[:nz, :]

def _shrink(X, r):
    """元素级软阈值（\ell1 近端）"""
    return np.sign(X) * np.maximum(np.abs(X) - r, 0.0)

def _solve_l1l2(X, tau):
    """
    \ell_{2,1} 近端：按列做 group shrinkage。
    对每一列 x:  y = max(0, 1 - tau/||x||_2) * x
    """
    Y = X.copy()
    norms = np.linalg.norm(Y, axis=0)  # 列范数
    # 避免除零
    scale = np.maximum(0.0, 1.0 - (tau / (norms + 1e-16)))
    Y *= scale
    return Y

def LRASR(X, Dict, beta, lamda, display=False,
          tol1=1e-6, tol2=1e-2, maxIter=100):
    """
    Python 版 LRASR:
        min ||S||_* + beta*||S||_1 + lamda*||E||_{2,1}
        s.t. X = Dict*S + E
    参数
    ----
    X : ndarray, shape (dim, num)
    Dict : ndarray, shape (dim, numDict)
    beta : float
    lamda : float
    display : bool
    tol1, tol2 : float
    maxIter : int

    返回
    ----
    S : ndarray, shape (numDict, num)
    E : ndarray, shape (dim, num)
    """
    X = np.asarray(X, dtype=np.float64)
    Dict = np.asarray(Dict, dtype=np.float64)

    dim, num = X.shape
    numDict = Dict.shape[1]

    # 初始化
    S = np.zeros((numDict, num))
    J = np.zeros_like(S)
    E = np.zeros_like(X)
    Y1 = np.zeros_like(X)
    Y2 = np.zeros_like(S)

    DtX = Dict.T @ X
    DtD = Dict.T @ Dict

    X_F = np.linalg.norm(X, ord='fro')

    # ADMM 参数
    mu = 0.01
    mu_max = 1e10
    # ita1 = 1 / ||Dict||_2^2
    smax = np.linalg.norm(Dict, 2)
    if smax > 0:
        ita1 = 1.0 / (smax * smax)
    else:
        ita1 = 1.0  # 退避，避免除零

    if display:
        # 近似秩（与 MATLAB rank(S, 1e-3*norm(S,2)) 类似）
        rS = np.linalg.matrix_rank(S, hermitian=False)
        print(f"initial, rank={rS}")

    for it in range(1, maxIter + 1):
        # ---- update S
        temp = (S
                + ita1 * DtX
                - ita1 * (DtD @ S)
                - ita1 * (Dict.T @ E)
                + ita1 * (Dict.T @ (Y1 / mu))
                - ita1 * S
                + ita1 * J
                - ita1 * (Y2 / mu))
        S1 = _svd_threshold(temp, ita1 / mu)

        # ---- update J (elementwise L1 prox)
        temp = S1 + (Y2 / mu)
        J1 = _shrink(temp, beta / mu)

        # ---- update E (L2,1 prox columnwise)
        temp = X - Dict @ S1 + (Y1 / mu)
        E1 = _solve_l1l2(temp, lamda / mu)

        # ---- duals
        RES = X - Dict @ S1 - E1
        Y1 = Y1 + mu * RES
        Y2 = Y2 + mu * (S1 - J1)

        # ---- mu 更新
        ktt2 = mu * max(
            max(np.sqrt(1.0 / ita1) * np.linalg.norm(S1 - S, ord='fro'),
                np.linalg.norm(J1 - J, ord='fro')),
            np.linalg.norm(E1 - E, ord='fro')
        ) / (X_F + 1e-16)
        rou = 1.1 if ktt2 < tol2 else 1.0
        mu = min(mu_max, rou * mu)

        # ---- 收敛判据
        ktt1 = np.linalg.norm(RES, ord='fro') / (X_F + 1e-16)

        S = S1
        J = J1
        E = E1

        if display:
            # 近似秩（用阈值 1e-3*||S||_2）
            s2 = np.linalg.norm(S, 2)
            thr = 1e-3 * s2
            # 用奇异值计数近似秩（避免 full svd 时可跳过）
            try:
                _, sing, _ = np.linalg.svd(S, full_matrices=False)
                rS = int(np.sum(sing > thr))
            except np.linalg.LinAlgError:
                rS = np.linalg.matrix_rank(S)
            print(f"iter {it}, mu={mu: .2e}, rank={rS}, "
                  f"stopC1={ktt1: .3e}, stopC2={ktt2: .3e}")

        if (ktt1 < tol1) and (ktt2 < tol2):
            break

    return S, E


# ---- quick test ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim, num, numDict = 50, 120, 80
    X = rng.standard_normal((dim, num))
    Dict = rng.standard_normal((dim, numDict))

    S, E = LRASR(X, Dict, beta=0.1, lamda=0.2, display=True, maxIter=50)
    print("S shape:", S.shape, "E shape:", E.shape,
          "recon err:", np.linalg.norm(X - Dict @ S - E) / np.linalg.norm(X))
