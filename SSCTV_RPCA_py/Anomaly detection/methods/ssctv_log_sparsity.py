from __future__ import annotations

import numpy as np
from numpy.fft import fftn, ifftn


# =========================
# 基础近端/差分算子
# =========================

def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """元素级软阈值。"""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _svt(mat: np.ndarray, tau: float) -> np.ndarray:
    """
    Singular Value Thresholding (SVT):
        argmin_X  ||X||_* + (1/2tau) ||X - A||_F^2
    在本实现里等价用于：
        argmin_X  ||X||_* + (mu/2) ||X - A||_F^2
    的闭式解，其中 tau = 1 / mu。
    """
    A = np.asarray(mat, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"_svt expects a 2D matrix, got shape={A.shape}")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_new = np.maximum(s - tau, 0.0)
    return (U * s_new) @ Vt


# -------- 前向差分（周期边界）与其伴随：负的后向差分 --------
def _diff_x(X_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(X_vec, (M, N, p), order="F")
    d = np.roll(cube, -1, axis=0) - cube
    return np.reshape(d, (M * N, p), order="F")


def _diff_y(X_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(X_vec, (M, N, p), order="F")
    d = np.roll(cube, -1, axis=1) - cube
    return np.reshape(d, (M * N, p), order="F")


def _diff_z(X_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(X_vec, (M, N, p), order="F")
    d = np.roll(cube, -1, axis=2) - cube
    return np.reshape(d, (M * N, p), order="F")


def _diff_xT(Y_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(Y_vec, (M, N, p), order="F")
    d = -(cube - np.roll(cube, 1, axis=0))
    return np.reshape(d, (M * N, p), order="F")


def _diff_yT(Y_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(Y_vec, (M, N, p), order="F")
    d = -(cube - np.roll(cube, 1, axis=1))
    return np.reshape(d, (M * N, p), order="F")


def _diff_zT(Y_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(Y_vec, (M, N, p), order="F")
    d = -(cube - np.roll(cube, 1, axis=2))
    return np.reshape(d, (M * N, p), order="F")


def _freq_energy(n: int) -> np.ndarray:
    """|FFT([1,-1])|^2 的采样（与 psf2otf([1,-1]) 的模方等价）。"""
    k = np.arange(n, dtype=np.float64)
    return 2.0 - 2.0 * np.cos(2.0 * np.pi * k / n)


# =========================
# Log-sparsity proximal
# =========================

def _log_sparse_objective_scalar(x: float, s: float, mu: float, lam: float) -> float:
    """
    标量目标：
        lam * log(1 + |x|) + (mu/2) * (x - s)^2
    """
    x = float(x)
    s = float(s)
    mu = float(mu)
    lam = float(lam)
    return lam * np.log1p(abs(x)) + 0.5 * mu * (x - s) * (x - s)


def _prox_log_sparse_scalar(s: float, mu: float, lam: float) -> float:
    """
    求解标量近端：
        argmin_x  lam * log(1 + |x|) + (mu/2) * (x - s)^2

    由于目标关于 x 的符号对称，最优解与 s 同号。
    令 u = |x|, v = |s|，则转化为：
        min_{u>=0} lam * log(1 + u) + (mu/2) * (u - v)^2

    一阶条件：
        lam/(1+u) + mu*(u-v) = 0
    化简为二次方程：
        u^2 + (1-v)u + (lam/mu - v) = 0

    为稳健起见，这里枚举：
        - 边界点 u = 0
        - 所有实非负驻点
    再比较目标函数值，选全局最优候选。
    """
    s = float(s)
    mu = float(max(mu, 1e-12))
    lam = float(max(lam, 0.0))

    if lam == 0.0:
        return s

    sign_s = 1.0 if s >= 0.0 else -1.0
    v = abs(s)

    candidates_u = [0.0, v]

    # u^2 + (1-v)u + (lam/mu - v) = 0
    a = 1.0
    b = 1.0 - v
    c = lam / mu - v

    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        sqrt_disc = np.sqrt(disc)
        u1 = (-b + sqrt_disc) / (2.0 * a)
        u2 = (-b - sqrt_disc) / (2.0 * a)
        if u1 >= 0.0:
            candidates_u.append(float(u1))
        if u2 >= 0.0:
            candidates_u.append(float(u2))

    # 去重
    uniq = []
    for u in candidates_u:
        if u < 0.0:
            continue
        if not any(abs(u - z) <= 1e-10 for z in uniq):
            uniq.append(float(u))

    best_x = 0.0
    best_val = _log_sparse_objective_scalar(0.0, s=s, mu=mu, lam=lam)
    for u in uniq:
        x = sign_s * u
        val = _log_sparse_objective_scalar(x, s=s, mu=mu, lam=lam)
        if val < best_val:
            best_val = val
            best_x = x

    return float(best_x)


def _log_sparse_shrink(mat: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """
    矩阵/张量逐元素 log-sparsity proximal：
        argmin_X  lam * sum log(1 + |X_ij|) + (mu/2) ||X - S||_F^2
    """
    A = np.asarray(mat, dtype=np.float64)
    flat = A.reshape(-1, order="F")
    out = np.empty_like(flat)

    for i, val in enumerate(flat):
        out[i] = _prox_log_sparse_scalar(float(val), mu=mu, lam=lam)

    return out.reshape(A.shape, order="F")


# =========================
# 主方法：SSCTV + log-sparsity sparse term
# =========================

def ssctv_log_sparsity(noise_data: np.ndarray, opts=None, verbose: bool = True):
    """
    SSCTV 的 log-sparsity 版本。

    对应目标：
        min_{X,S}  ||G31||_* + ||G32||_*
                 + lambda * sum_{i,j} log(1 + |S_{ij}|)

    s.t.
        Y = X + S,
        Dx(X) = G1,
        Dy(X) = G2,
        Dz(G1) = G31,
        Dz(G2) = G32.

    其中稀疏项把原始 SSCTV 的 L1:
        lambda * ||S||_1
    替换为：
        lambda * sum log(1 + |S|)

    参数
    ----
    noise_data : np.ndarray
        输入高光谱立方体，shape=(M,N,p)
    opts : dict
        可选参数：
        - maxIter: 最大迭代次数，默认 1000
        - rho: 罚参数增长率，默认 1.03
        - tol: 收敛阈值，默认 1e-6
        - lambda / lambda_ / lambdaVal / lambda_val: 稀疏权重
        - mu, mu1: 初始罚参数；不传则自动初始化

    返回
    ----
    output_image : np.ndarray
        恢复结果 X，shape=(M,N,p)
    S : np.ndarray
        稀疏/残差项，shape=(M*N,p)

    备注
    ----
    - 返回接口故意保持成和原 ssctv_rpca.py / ssctv_rpca_logdet.py 一致，便于 test.py 直接调用。
    - 这个方法对应“原 SSCTV 的核范数结构项保持不变，只把 S 上的 L1 换成 log-sparsity”。
    """
    if opts is None:
        opts = {}

    M, N, p = noise_data.shape
    sizeD = (M, N, p)

    maxIter = int(opts.get("maxIter", 1000))
    rho = float(opts.get("rho", 1.03))
    tol = float(opts.get("tol", 1e-6))

    lam = float(
        opts.get(
            "lambda",
            opts.get(
                "lambda_",
                opts.get(
                    "lambdaVal",
                    opts.get(
                        "lambda_val",
                        2.0 / np.sqrt(M * N)
                    ),
                ),
            ),
        )
    )

    # D: (M*N, p) —— 列主序，和原实现保持一致
    D = np.reshape(noise_data.astype(np.float64), (M * N, p), order="F")
    normD = np.linalg.norm(D, ord="fro")

    # dual norm 初始化，沿用 RPCA/SSCTV 的常见做法
    smax = np.linalg.svd(D, compute_uv=False)[0]
    norm_two = smax
    norm_inf = np.linalg.norm(D.ravel(order="F"), ord=np.inf) / max(lam, 1e-12)
    dual_norm = max(norm_two, norm_inf)

    mu = float(opts.get("mu", 1.0 / max(dual_norm, 1e-12)))
    mu1 = float(opts.get("mu1", mu))
    max_mu = max(mu, mu1) * 1e7

    # 频域能量项（与原 SSCTV 相同）
    Enx = _freq_energy(M)[:, None, None]
    Eny = _freq_energy(N)[None, :, None]
    Enz = _freq_energy(p)[None, None, :]
    determ_xy = Enx + Eny
    determ_z = Enz

    # 初始化变量
    X = D.copy()
    G1 = np.zeros_like(D)
    G2 = np.zeros_like(D)
    G31 = np.zeros_like(D)
    G32 = np.zeros_like(D)
    S = np.zeros_like(D)

    M1 = D / max(dual_norm, 1e-12)
    M2 = M1.copy()
    M3 = M1.copy()
    M4 = M1.copy()
    M5 = M1.copy()

    for it in range(1, maxIter + 1):
        # -------------------------
        # 更新 G31, G32：核范数 prox（原 SSCTV）
        # -------------------------
        A1 = np.reshape(_diff_z(G1, sizeD), (M * N, p), order="F") + M4 / mu1
        G31 = _svt(A1, tau=1.0 / max(mu1, 1e-12))

        A2 = np.reshape(_diff_z(G2, sizeD), (M * N, p), order="F") + M5 / mu1
        G32 = _svt(A2, tau=1.0 / max(mu1, 1e-12))

        # -------------------------
        # 更新 G1
        # min_G1 (mu1/2)||Dz G1 - G31 + M4/mu1||^2 + (mu1/2)||Dx X - G1 + M2/mu1||^2
        # -------------------------
        diffT_p = _diff_zT(mu1 * G31 - M4, sizeD)
        numer = diffT_p + mu1 * _diff_x(X, sizeD) + M2
        num_cube = np.reshape(numer, (M, N, p), order="F")
        den = mu1 * determ_z + mu1
        x = np.real(ifftn(fftn(num_cube) / den))
        G1 = np.reshape(x, (M * N, p), order="F")

        # -------------------------
        # 更新 G2
        # -------------------------
        diffT_p = _diff_zT(mu1 * G32 - M5, sizeD)
        numer = diffT_p + mu1 * _diff_y(X, sizeD) + M3
        num_cube = np.reshape(numer, (M, N, p), order="F")
        den = mu1 * determ_z + mu1
        x = np.real(ifftn(fftn(num_cube) / den))
        G2 = np.reshape(x, (M * N, p), order="F")

        # -------------------------
        # 更新 X
        # min_X (mu/2)||D-X-S+M1/mu||^2
        #     + (mu1/2)||Dx X-G1+M2/mu1||^2
        #     + (mu1/2)||Dy X-G2+M3/mu1||^2
        # -------------------------
        diffT_p = _diff_xT(mu1 * G1 - M2, sizeD) + _diff_yT(mu1 * G2 - M3, sizeD)
        numer = diffT_p + mu * (D - S) + M1
        num_cube = np.reshape(numer, (M, N, p), order="F")
        den = mu1 * determ_xy + mu
        x = np.real(ifftn(fftn(num_cube) / den))
        X = np.reshape(x, (M * N, p), order="F")

        # -------------------------
        # 更新 S：log-sparsity prox
        # min_S lambda * sum log(1 + |S_ij|) + (mu/2)||D-X-S+M1/mu||^2
        # -------------------------
        Z = D - X + M1 / mu
        S = _log_sparse_shrink(Z, lam=lam, mu=mu)

        # -------------------------
        # 计算残差
        # -------------------------
        leq1 = D - X - S
        leq2 = np.reshape(_diff_x(X, sizeD), (M * N, p), order="F") - G1
        leq3 = np.reshape(_diff_y(X, sizeD), (M * N, p), order="F") - G2
        leq4 = np.reshape(_diff_z(G1, sizeD), (M * N, p), order="F") - G31
        leq5 = np.reshape(_diff_z(G2, sizeD), (M * N, p), order="F") - G32

        stopC1 = np.linalg.norm(leq1, ord="fro") / (normD + 1e-12)
        stopC2 = np.max(np.abs(leq2))
        stopC3 = np.max(np.abs(leq3))
        stopC4 = np.linalg.norm(leq4, ord="fro") / (normD + 1e-12)
        stopC5 = np.linalg.norm(leq5, ord="fro") / (normD + 1e-12)
        stop_all = max(stopC1, stopC2, stopC3, stopC4, stopC5)

        if verbose and (it % 10 == 0 or it == 1):
            print(
                f"iter {it:3d}, mu={mu:8.2e}, mu1={mu1:8.2e}, "
                f"r1={stopC1:8.2e}, r2={stopC2:8.2e}, r3={stopC3:8.2e}, "
                f"r4={stopC4:8.2e}, r5={stopC5:8.2e}"
            )

        if stop_all < tol:
            break

        # -------------------------
        # 更新乘子
        # -------------------------
        M1 = M1 + mu * leq1
        M2 = M2 + mu1 * leq2
        M3 = M3 + mu1 * leq3
        M4 = M4 + mu1 * leq4
        M5 = M5 + mu1 * leq5

        mu = min(max_mu, mu * rho)
        mu1 = min(max_mu, mu1 * rho)

    output_image = np.reshape(X, (M, N, p), order="F")
    return output_image, S


# 兼容可能的导入方式
ssctv_rpca_log_sparsity = ssctv_log_sparsity
ssctv_logsparsity = ssctv_log_sparsity


if __name__ == "__main__":
    # 简单自测
    np.random.seed(0)
    cube = np.random.rand(20, 18, 8)
    opts = {
        "maxIter": 5,
        "rho": 1.03,
        "tol": 1e-5,
        "lambda": 2.0 / np.sqrt(20 * 18),
    }
    X, S = ssctv_log_sparsity(cube, opts=opts, verbose=True)
    print("X shape:", X.shape)
    print("S shape:", S.shape)