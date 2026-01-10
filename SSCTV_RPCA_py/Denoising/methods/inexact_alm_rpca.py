# inexact_alm_rpca.py
# -*- coding: utf-8 -*-
import numpy as np

try:
    # 可选：若安装了 SciPy，用 svds 做部分 SVD（大矩阵更快）
    from scipy.sparse.linalg import svds as _svds
    _HAS_SVDS = True
except Exception:
    _HAS_SVDS = False


def _soft_threshold(X, tau):
    """元素级软阈值：sign(x)*max(|x|-tau,0)"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def inexact_alm_rpca(D, lambda_=None, tol=1e-7, maxIter=1000, verbose=False):
    """
    Robust PCA via Inexact Augmented Lagrange Multiplier (IALM)
    目标：min ||A||_* + lambda * ||E||_1  s.t. D = A + E

    参数
    ----
    D : (m, n) ndarray
        观测矩阵
    lambda_ : float, 默认与 MATLAB 版本一致为 1/sqrt(m)
    tol : float, 停止阈值，默认 1e-7
    maxIter : int, 最大迭代次数，默认 1000
    verbose : bool, 是否打印进度

    返回
    ----
    A_hat : 低秩部分
    E_hat : 稀疏部分
    iter  : 实际迭代次数
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    if lambda_ is None:
        # 与你贴的 MATLAB 代码保持一致：lambda = 1/sqrt(m)
        lambda_ = 1.0 / np.sqrt(m)

    if tol is None or tol == -1:
        tol = 1e-7
    if maxIter is None or maxIter == -1:
        maxIter = 1000

    # 初始化 Y，使其满足 ||Y||_2 <= 1, ||Y||_inf <= 1/lambda
    # MATLAB: norm_two = lansvd(Y,1,'L'); 这里用谱范数代替
    norm_two = np.linalg.norm(D, 2)
    norm_inf = np.max(np.abs(D)) / lambda_
    dual_norm = max(norm_two, norm_inf)
    Y = D / dual_norm

    A_hat = np.zeros((m, n), dtype=np.float64)
    E_hat = np.zeros((m, n), dtype=np.float64)

    # MATLAB: mu = 1.25/norm_two; mu_bar = mu*1e7; rho = 1.5
    mu = 1.25 / (norm_two + 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    d_norm = np.linalg.norm(D, 'fro')

    iter_ = 0
    total_svd = 0
    converged = False

    # 初始“期望秩”参数（模仿 choosvd 的增减逻辑）
    sv = 10

    while not converged and iter_ < maxIter:
        iter_ += 1

        # ---- 更新 E：element-wise 软阈值 ----
        temp_T = D - A_hat + (1.0 / mu) * Y
        E_hat = _soft_threshold(temp_T, lambda_ / mu)

        # ---- 更新 A：奇异值软阈值（SVT）----
        M = D - E_hat + (1.0 / mu) * Y

        # 选择 full SVD 或部分 SVD（模仿 choosvd 的启发式）
        use_partial = False
        k = sv
        if _HAS_SVDS:
            mn = min(m, n)
            # 启发式：矩阵较大且期望秩 sv 远小于维度时用部分 SVD
            if mn > 200 and 1 < k < mn - 1:
                use_partial = True
                k = min(k, mn - 1)

        if use_partial:
            # SciPy svds 返回无序奇异值/向量，这里要从大到小排序
            # 只取最大的 k 个奇异值（与 lansvd 一致）
            try:
                U, s, Vt = _svds(M, k=k)   # s 升序或无序，需要排序
                idx = np.argsort(s)[::-1]
                s = s[idx]
                U = U[:, idx]
                Vt = Vt[idx, :]
            except Exception:
                # 退回 full SVD
                U, s, Vt = np.linalg.svd(M, full_matrices=False)
                use_partial = False
        else:
            U, s, Vt = np.linalg.svd(M, full_matrices=False)

        total_svd += 1

        # 奇异值软阈值：sigma' = max(sigma - 1/mu, 0)
        thresh = 1.0 / mu
        s_shrunk = np.maximum(s - thresh, 0.0)
        svp = int((s_shrunk > 0).sum())

        if svp >= 1:
            A_hat = (U[:, :svp] * s_shrunk[:svp]) @ Vt[:svp, :]
        else:
            A_hat = np.zeros_like(M)

        # 动态调整下一轮的期望秩 sv（与 MATLAB 逻辑一致）
        if svp < sv:
            sv = min(svp + 1, min(m, n))
        else:
            sv = min(svp + int(round(0.05 * min(m, n))), min(m, n))

        # ---- 对偶更新 ----
        Z = D - A_hat - E_hat
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        # ---- 停止准则 ----
        stopCriterion = np.linalg.norm(Z, 'fro') / (d_norm + 1e-12)
        if verbose and (total_svd % 10 == 0):
            # 近似 rank(A_hat) 的打印：svp 就是当前软阈后非零奇异值个数
            nnz_E = int((np.abs(E_hat) > 0).sum())
            print(f"#svd {total_svd:4d}  r(A) {svp:3d}  |E|_0 {nnz_E:7d}  stop {stopCriterion:.3e}")

        if stopCriterion < tol:
            converged = True

    if not converged and verbose:
        print("Maximum iterations reached.")

    return A_hat, E_hat, iter_
