# unsupervised_rpca_detect_v1.py
# -*- coding: utf-8 -*-
import numpy as np

def _soft_threshold(X, tau):
    """elementwise soft-thresholding."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

def inexact_alm_rpca(M, lambda_=None, tol=1e-7, max_iter=1000, rho=1.5):
    """
    Robust PCA via Inexact Augmented Lagrangian Multiplier (IALM).
    Solve:  min ||L||_* + lambda * ||S||_1  s.t. M = L + S

    Parameters
    ----------
    M : ndarray, shape (m, n)
    lambda_ : float or None
        If None, use 1/sqrt(max(m,n))
    tol : float
        Relative residual tolerance
    max_iter : int
    rho : float
        mu growth rate

    Returns
    -------
    L : ndarray, low-rank
    S : ndarray, sparse
    it : int, iterations
    """
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape
    if lambda_ is None:
        lambda_ = 1.0 / np.sqrt(max(m, n))

    # Initialize
    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.linalg.norm(M, np.inf) / lambda_
    dual_norm = max(norm_two, norm_inf)
    Y = M / dual_norm

    mu = 1.25 / norm_two if norm_two > 0 else 1.25
    mu_bar = mu * 1e7

    L = np.zeros_like(M)
    S = np.zeros_like(M)

    froM = np.linalg.norm(M, 'fro') + 1e-16

    for it in range(1, max_iter + 1):
        # --- update L by singular value thresholding ---
        U, s, Vt = np.linalg.svd(M - S + (1.0 / mu) * Y, full_matrices=False)
        s_shrink = np.maximum(s - 1.0 / mu, 0.0)
        r = np.sum(s_shrink > 0.0)
        if r == 0:
            L[:] = 0.0
        else:
            L = (U[:, :r] * s_shrink[:r]) @ Vt[:r, :]

        # --- update S by elementwise soft-thresholding ---
        S = _soft_threshold(M - L + (1.0 / mu) * Y, lambda_ / mu)

        # --- dual & stopping ---
        R = M - L - S
        err = np.linalg.norm(R, 'fro') / froM
        if err < tol:
            return L, S, it

        Y = Y + mu * R
        mu = min(mu * rho, mu_bar)

    return L, S, it

def Unsupervised_RPCA_Detect_v1(Data, lambda_=None, compute_result=True):
    """
    Python 等价版:
      [result, Data_tmp1, Data_tmp2] = Unsupervised_RPCA_Detect_v1(Data, lambda)

    参数
    ----
    Data : ndarray, shape (a, b, c)
        高光谱立方体（行×列×光谱）
    lambda_ : float or None
        RPCA 稀疏项权重（None 时采用 1/sqrt(max(a*b, c))）
    compute_result : bool
        是否返回按光谱维的 L1 异常强度图（与 MATLAB 注释代码一致）

    返回
    ----
    result : ndarray, shape (a, b)
        每像素在稀疏项 S 上的 L1 范数；若 compute_result=False，则为全零
    Data_tmp1 : ndarray, shape (a, b, c)
        稀疏项 S 的三维重排
    Data_tmp2 : ndarray, shape (a, b, c)
        低秩项 L 的三维重排
    """
    a, b, c = Data.shape
    # MATLAB: DataTest = reshape(Data, a*b, c);
    DataTest = Data.reshape(a * b, c, order='F')

    # RPCA
    L_hat, S_hat, _ = inexact_alm_rpca(DataTest, lambda_=lambda_)

    # 回到 3D（与 MATLAB 一致，用 Fortran 顺序）
    Data_tmp1 = S_hat.reshape(a, b, c, order='F')
    Data_tmp2 = L_hat.reshape(a, b, c, order='F')

    if compute_result:
        # 每像素沿光谱维的 L1 范数（等价于 MATLAB 中被注释的循环）
        result = np.sum(np.abs(Data_tmp1), axis=2)
    else:
        result = np.zeros((a, b), dtype=np.float64)

    return result, Data_tmp1, Data_tmp2


# ---- quick test ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a, b, c = 30, 25, 10
    L0 = rng.standard_normal((a*b, 3)) @ rng.standard_normal((3, c))
    S0 = np.zeros((a*b, c))
    idx = rng.choice(a*b*c, size=500, replace=False)
    S0.reshape(-1)[idx] = rng.standard_normal(500) * 3
    M = L0 + S0
    Data = M.reshape(a, b, c, order='F')

    result, S3, L3 = Unsupervised_RPCA_Detect_v1(Data, lambda_=1/np.sqrt(max(a*b, c)))
    print(result.shape, S3.shape, L3.shape)
