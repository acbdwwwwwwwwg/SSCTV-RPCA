# lrr_tv_manifold.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from construct_w import constructW  # 确保文件在你的 PYTHONPATH 下

def soft(X, tau):
    """元素级 soft-threshold."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)

def solve_l1l2(W, tau):
    """
    列分组 L2,1 近端：对每一列 w，y = max(0, 1 - tau/||w||_2)*w
    """
    E = W.copy()
    norms = np.linalg.norm(E, axis=0)
    scale = np.maximum(0.0, 1.0 - (tau / (norms + 1e-16)))
    E *= scale
    return E

def _pcg_numpy(Aop, b, x0=None, tol=1e-6, maxiter=1000):
    """
    纯 NumPy 的共轭梯度（A 对称正定），Aop 可以是 ndarray 或可调用 mv(x).
    """
    if callable(Aop):
        mv = Aop
    else:
        A = Aop
        mv = lambda x: A @ x

    n = b.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - mv(x)
    p = r.copy()
    rs_old = r @ r
    if rs_old < tol**2:
        return x
    for _ in range(maxiter):
        Ap = mv(p)
        alpha = rs_old / (p @ Ap + 1e-30)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if rs_new < tol**2:
            break
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new
    return x

def lrr_tv_manifold(Y, A, lambda_, beta, gamma, im_size, display=True,
                    maxIter=400, mu_init=1e-4, mu_bar=1e10, rho=1.5,
                    pcg_tol=1e-6, pcg_maxiter=200):
    """
    Python 等价版：
        min ||X||_* + lambda*||S||_{2,1} + beta*||H(X)||_{1,1} + gamma*tr(X L X^T)
        s.t. Y = A X + S

    参数
    ----
    Y : ndarray, shape (L, N)
    A : ndarray, shape (L, m)
    lambda_, beta, gamma : float
    im_size : (H, W)，要求 H*W == N
    display : bool
    其余参数：与 MATLAB 基本对应

    返回
    ----
    X : ndarray, shape (m, N)
    S : ndarray, shape (L, N)
    """
    Y = np.asarray(Y, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    L, N = Y.shape
    m = A.shape[1]
    H, W = int(im_size[0]), int(im_size[1])
    assert H * W == N, "im_size 与 Y 的列数 N 不一致"

    # ---------- 图拉普拉斯 ----------
    opts = {'NeighborMode': 'KNN', 'k': 5, 'WeightMode': 'HeatKernel', 't': 1.0}
    Wg = constructW(Y.T, opts)  # 构造在列（样本）上的图
    if _HAS_SCIPY:
        if not sp.issparse(Wg):
            Wg = sp.csr_matrix(Wg)
        d = np.asarray(Wg.sum(axis=1)).ravel()
        Lap = sp.diags(d) - Wg
    else:
        if Wg.ndim == 2:
            d = Wg.sum(axis=1)
            Lap = np.diag(d) - Wg
        else:
            raise RuntimeError("Need dense Laplacian when SciPy is unavailable.")

    # ---------- FFT 周期差分算子（仅在 HxW 平面上） ----------
    FDh = np.zeros((H, W), dtype=np.complex128)
    FDh[0, 0] = -1.0
    FDh[0, -1] = 1.0
    FDh = np.fft.fft2(FDh)
    FDhH = FDh.conj()

    FDv = np.zeros((H, W), dtype=np.complex128)
    FDv[0, 0] = -1.0
    FDv[-1, 0] = 1.0
    FDv = np.fft.fft2(FDv)
    FDvH = FDv.conj()

    IL = 1.0 / (FDhH * FDh + FDvH * FDv + 1.0)  # 逐频域的“逆滤波器”

    def Dh(x2d):  # (H,W) -> (H,W)
        return np.real(np.fft.ifft2(np.fft.fft2(x2d) * FDh))

    def Dv(x2d):
        return np.real(np.fft.ifft2(np.fft.fft2(x2d) * FDv))

    def DhH_(x2d):
        return np.real(np.fft.ifft2(np.fft.fft2(x2d) * FDhH))

    def DvH_(x2d):
        return np.real(np.fft.ifft2(np.fft.fft2(x2d) * FDvH))

    # ---------- 初始化 ----------
    X = np.zeros((m, N), dtype=np.float64)  # X0 == 0
    S = np.zeros((L, N), dtype=np.float64)

    # Lagrange 与 V 变量
    V1 = X.copy()          # 核范数项近端的变量
    V2 = X.copy()          # 流形项近端的变量
    V3 = X.copy()          # TV 重建的变量（矩阵形式）
    # V4 是每个 band 的 (horiz, vert) 差分图
    V4_h = [None] * m
    V4_v = [None] * m

    D1 = np.zeros_like(Y)  # data constraint 的乘子
    D2 = np.zeros_like(X)  # 对应 V1
    D3 = np.zeros_like(X)  # 对应 V2
    D4 = np.zeros_like(X)  # 对应 V3
    # D5 是对 (horiz, vert) 的乘子
    D5_h = [np.zeros((H, W), dtype=np.float64) for _ in range(m)]
    D5_v = [np.zeros((H, W), dtype=np.float64) for _ in range(m)]

    # 预计算 (A^T A + 3 I)^{-1}
    K = A.T @ A + 3.0 * np.eye(m, dtype=np.float64)
    K_inv = np.linalg.inv(K)

    mu = mu_init
    tol = np.sqrt(N) * 1e-5

    # ---------- 迭代 ----------
    for it in range(1, maxIter + 1):

        # --- 1) 更新 X：解 (A^T A + 3I) X = RHS
        Xi = A.T @ (Y - D1 - S)
        for Vj, Dj in ((V1, D2), (V2, D3), (V3, D4)):
            Xi += (Vj - Dj)
        X = K_inv @ Xi  # (m,N)

        # --- 2) 近端们：V1 (核范数), V2 (图拉普拉斯), V3/V4 (TV)
        # V1：SVT 阈值 1/mu
        temp = X + D2
        U, s, Vt = np.linalg.svd(temp, full_matrices=False)
        s_shrink = np.maximum(s - 1.0 / mu, 0.0)
        r = np.count_nonzero(s_shrink > 0)
        V1 = (U[:, :r] * s_shrink[:r]) @ Vt[:r, :] if r > 0 else np.zeros_like(temp)

        # V2：对每个通道 i，解 (2γL + μI) v_i^T = μ (x_i + d3_i)^T
        coef = 2.0 * gamma * Lap
        def Aop_vec(v):
            # v is (N,)
            if _HAS_SCIPY:
                return (coef @ v) + mu * v
            else:
                return (coef @ v) + mu * v  # coef 为 dense ndarray 时也 OK

        temp2 = (mu * (X + D3)).T  # (N, m)
        M = np.zeros_like(temp2)
        if _HAS_SCIPY:
            # 用 scipy.sparse.linalg.cg
            for i in range(m):
                Mi, info = spla.cg(coef + mu * (sp.eye(N) if _HAS_SCIPY else np.eye(N)),
                                   temp2[:, i], tol=pcg_tol, maxiter=pcg_maxiter)
                M[:, i] = Mi
        else:
            # 纯 numpy PCG
            for i in range(m):
                M[:, i] = _pcg_numpy(Aop_vec, temp2[:, i], tol=pcg_tol, maxiter=pcg_maxiter)
        V2 = M.T  # (m,N)

        # V3/V4：TV 近端
        nu_aux = X + D4  # (m,N)
        # 转成 (H,W,m) —— 注意与 MATLAB 对齐使用 'F' 顺序
        nu_aux_im = nu_aux.T.reshape((H, W, m), order='F')

        V3_im = np.zeros((H, W, m), dtype=np.float64)
        for k in range(m):
            rhs = DhH_( (V4_h[k] if V4_h[k] is not None else np.zeros((H,W))) - D5_h[k] ) + \
                  DvH_( (V4_v[k] if V4_v[k] is not None else np.zeros((H,W))) - D5_v[k] ) + \
                  nu_aux_im[:, :, k]
            V3_im[:, :, k] = np.real(np.fft.ifft2(IL * np.fft.fft2(rhs)))

            # V4：对梯度做 soft
            aux_h = Dh(V3_im[:, :, k])
            aux_v = Dv(V3_im[:, :, k])
            V4_h[k] = soft(aux_h + D5_h[k], beta / mu)
            V4_v[k] = soft(aux_v + D5_v[k], beta / mu)

            # 更新 D5（梯度的乘子）
            D5_h[k] = D5_h[k] + (aux_h - V4_h[k])
            D5_v[k] = D5_v[k] + (aux_v - V4_v[k])

        # V3 从 (H,W,m) 回到 (m,N)
        V3 = V3_im.reshape((H * W, m), order='F').T

        # --- 3) 更新 S ：L2,1 近端
        S = solve_l1l2(Y - A @ X - D1, lambda_ / mu)

        # --- 4) 更新乘子 D1..D4
        D1 = D1 - (Y - A @ X - S)
        D2 = D2 + (X - V1)
        D3 = D3 + (X - V2)
        D4 = D4 + (X - V3)

        # --- 5) 监控与收敛
        if it == 1 or it % 10 == 1:
            res = np.zeros(4, dtype=np.float64)
            res[0] = np.linalg.norm(Y - A @ X - S, ord='fro')
            res[1] = np.linalg.norm(X - V1, ord='fro')
            res[2] = np.linalg.norm(X - V2, ord='fro')
            res[3] = np.linalg.norm(X - V3, ord='fro')
            if display:
                print(f"iter={it:3d}  res(1..4)= "
                      f"{res[0]:.6e}, {res[1]:.6e}, {res[2]:.6e}, {res[3]:.6e}")
            if np.sum(np.abs(res)) <= tol:
                break

        mu = min(mu * rho, mu_bar)

    return X, S


# ------------- quick test -------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W, m, L = 20, 18, 6, 8
    N = H * W
    # 合成数据：A (L×m), X (m×N), S (L×N)
    A = rng.standard_normal((L, m))
    X0 = rng.standard_normal((m, N))
    S0 = np.zeros((L, N))
    idx = rng.choice(L*N, size=300, replace=False)
    S0.reshape(-1)[idx] = rng.standard_normal(300) * 0.5
    Y = A @ X0 + S0 + 0.05 * rng.standard_normal((L, N))

    X, S = lrr_tv_manifold(Y, A, lambda_=0.1, beta=0.05, gamma=1e-2,
                           im_size=(H, W), display=True, maxIter=50)
    print("X,S shapes:", X.shape, S.shape,
          "| data residual:", np.linalg.norm(Y - A @ X - S) / np.linalg.norm(Y))
