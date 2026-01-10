# methods/ctv_rpca.py
# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fftn, ifftn


def _psf2otf(psf, out_shape):
    """把差分核转为与 out_shape 相同大小的 OTF（周期边界）"""
    psf_pad = np.zeros(out_shape, dtype=np.float64)
    # 将 PSF 放到左上角
    idx = tuple(slice(0, s) for s in psf.shape)
    psf_pad[idx] = psf
    # 循环移动，使“中心”到 (0,0,0)
    for axis, (ps, os) in enumerate(zip(psf.shape, out_shape)):
        psf_pad = np.roll(psf_pad, -int(ps // 2), axis=axis)
    otf = fftn(psf_pad)
    return otf


def _reshape_to_cube(A, sizeD):
    # MATLAB 的(:)是列主序，这里要保持一致
    return np.reshape(A, sizeD, order="F")


def _reshape_to_mat(cube):
    h, w, d = cube.shape
    return np.reshape(cube, (h * w, d), order="F")


def _diff_x(mat, sizeD):
    X = _reshape_to_cube(mat, sizeD)
    return X - np.roll(X, -1, axis=0)  # forward diff (周期边界)


def _diff_y(mat, sizeD):
    X = _reshape_to_cube(mat, sizeD)
    return X - np.roll(X, -1, axis=1)


def _diff_z(mat, sizeD):
    X = _reshape_to_cube(mat, sizeD)
    return X - np.roll(X, -1, axis=2)


def _diff_xT(mat, sizeD):
    G = _reshape_to_cube(mat, sizeD)
    div = G - np.roll(G, 1, axis=0)    # adjoint of forward diff
    return _reshape_to_mat(div)


def _diff_yT(mat, sizeD):
    G = _reshape_to_cube(mat, sizeD)
    div = G - np.roll(G, 1, axis=1)
    return _reshape_to_mat(div)


def _diff_zT(mat, sizeD):
    G = _reshape_to_cube(mat, sizeD)
    div = G - np.roll(G, 1, axis=2)
    return _reshape_to_mat(div)


def _soft_thresh(X, t):
    return np.sign(X) * np.maximum(np.abs(X) - t, 0.0)


def _svd_shrink(M, tau):
    """奇异值软阈值化：U * shrink(S) * V^T"""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S = np.maximum(S - tau, 0.0)
    return (U * S) @ Vt


def ctv_rpca(oriData3_noise, opts=None):
    """
    min_X ||D_x(X)||_* + ||D_y(X)||_* + ||D_z(X)||_* + lambda * ||E||_1
    s.t.  Y = X + E

    Parameters
    ----------
    oriData3_noise : np.ndarray, shape (M, N, p)
    opts : dict, keys = {'maxIter','rho','tol','lambda','weight'}

    Returns
    -------
    output_image : np.ndarray, shape (M, N, p)
        恢复的 X
    E : np.ndarray, shape (M*N, p)
        稀疏噪声（与 MATLAB 保持同形状返回）
    """
    if opts is None:
        opts = {}

    M, N, p = oriData3_noise.shape
    sizeD = (M, N, p)

    maxIter = int(opts.get("maxIter", 200))
    rho = float(opts.get("rho", 1.25))
    tol = float(opts.get("tol", 1e-6))
    lam = float(opts.get("lambda", 3.0 / np.sqrt(M * N)))
    weight = float(opts.get("weight", 1.0))

    # 矩阵化 D（列主序）
    D = _reshape_to_mat(oriData3_noise)

    normD = np.linalg.norm(D, "fro")
    # 估最大奇异值
    norm_two = np.linalg.svd(D, compute_uv=False)[0]
    norm_inf = np.max(np.abs(D)) / max(lam, 1e-12)
    dual_norm = max(norm_two, norm_inf)

    mu = 1.25 / dual_norm
    mu1 = 0.25 * mu
    max_mu = mu * 1e7

    # === OTF（频域中的差分核能量）===
    h, w, d = sizeD
    psf_x = np.zeros((2, 1, 1)); psf_x[0, 0, 0] = 1; psf_x[1, 0, 0] = -1
    psf_y = np.zeros((1, 2, 1)); psf_y[0, 0, 0] = 1; psf_y[0, 1, 0] = -1
    psf_z = np.zeros((1, 1, 2)); psf_z[0, 0, 0] = 1; psf_z[0, 0, 1] = -1

    Eny_x = np.abs(_psf2otf(psf_x, sizeD)) ** 2
    Eny_y = np.abs(_psf2otf(psf_y, sizeD)) ** 2
    Eny_z = np.abs(_psf2otf(psf_z, sizeD)) ** 2
    determ = Eny_x + Eny_y + Eny_z

    # === 初始化 ===
    X = D.copy()
    E = np.zeros_like(D)

    # 乘子（按原 m 代码的共享初始化）
    M1 = D / dual_norm
    M2 = M1.copy()
    M3 = M1.copy()
    M4 = M1.copy()

    for it in range(1, maxIter + 1):
        # --- 更新 X1, X2, X3（逐项做核范数阈值/SVT）---
        tmp = _reshape_to_mat(_diff_x(X, sizeD)) + M2 / mu1
        X1 = _svd_shrink(tmp, 1.0 / mu1)

        tmp = _reshape_to_mat(_diff_y(X, sizeD)) + M3 / mu1
        X2 = _svd_shrink(tmp, 1.0 / mu1)

        tmp = _reshape_to_mat(_diff_z(X, sizeD)) + M4 / mu1
        X3 = _svd_shrink(tmp, weight / mu1)

        # --- 更新 X（频域闭式解）---
        diffT_p = _diff_xT(mu1 * X1 - M2, sizeD) \
                + _diff_yT(mu1 * X2 - M3, sizeD) \
                + _diff_zT(mu1 * X3 - M4, sizeD)

        numer1 = _reshape_to_cube(diffT_p + mu * (D - E) + M1, sizeD)
        x = np.real(ifftn(fftn(numer1) / (mu1 * determ + mu)))
        X = _reshape_to_mat(x)

        # --- 更新 E（逐元素软阈值）---
        E = _soft_thresh(D - X + M1 / mu, lam / mu)

        # --- 终止准则 ---
        leq1 = D - X - E
        leq2 = _reshape_to_mat(_diff_x(X, sizeD)) - X1
        leq3 = _reshape_to_mat(_diff_y(X, sizeD)) - X2
        leq4 = _reshape_to_mat(_diff_z(X, sizeD)) - X3

        stopC1 = np.linalg.norm(leq1, "fro") / max(normD, 1e-12)
        stopC2 = np.max(np.abs(leq2))
        stopC4 = np.linalg.norm(leq4, "fro") / max(normD, 1e-12)

        # 可选：打印
        if it % 10 == 0 or it == 1:
            print(f"iter {it:3d}, mu={mu:.2e}, Y-X-E={stopC1:.3e}, "
                  f"||DX-X1||_inf={stopC2:.3e}, |DZ-X3|_F/normD={stopC4:.3e}")

        if stopC1 < tol and stopC2 < tol:
            break

        # --- 更新乘子 & 惩罚参数 ---
        M1 = M1 + mu * leq1
        M2 = M2 + mu1 * leq2
        M3 = M3 + mu1 * leq3
        M4 = M4 + mu1 * leq4

        mu = min(max_mu, mu * rho)
        mu1 = min(max_mu, mu1 * rho)

    output_image = _reshape_to_cube(X, sizeD)
    return output_image, E
