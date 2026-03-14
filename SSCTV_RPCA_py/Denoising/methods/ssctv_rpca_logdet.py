import numpy as np
from numpy.fft import fftn, ifftn


# =========================
# 基础近端/差分算子
# =========================

def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """元素级软阈值。"""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _elastic_net_shrink(x: np.ndarray, lambda1: float, lambda2: float, mu: float) -> np.ndarray:
    """
    求解
        argmin_s  2*lambda1*|s| + lambda2*s^2 + (mu/2)*(s-x)^2
    的闭式解（逐元素）。

    对应整矩阵形式：
        argmin_S  2*lambda1*||S||_1 + lambda2*||S||_F^2 + (mu/2)||S-X||_F^2

    闭式：
        S = soft(X, 2*lambda1/mu) / (1 + 2*lambda2/mu)
    """
    tau = 2.0 * lambda1 / max(mu, 1e-12)
    scale = 1.0 + 2.0 * lambda2 / max(mu, 1e-12)
    return _soft_threshold(x, tau) / scale


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
# LogDet proximal（内嵌实现）
# =========================

def _logdet_objective_scalar(x: float, s: float, mu: float, alpha: float) -> float:
    x = float(max(x, 0.0))
    s = float(max(s, 0.0))
    mu = float(mu)
    alpha = float(alpha)
    return alpha * np.log1p(x * x) + 0.5 * mu * (x - s) * (x - s)


def _logdet_roots(s: float, mu: float, alpha: float, imag_tol: float = 1e-9) -> np.ndarray:
    """
    对标量子问题
        min_{x>=0} alpha*log(1+x^2) + (mu/2)*(x-s)^2
    的一阶条件求根：
        mu*x^3 - mu*s*x^2 + (mu + 2*alpha)*x - mu*s = 0
    """
    coeff = np.array([
        mu,
        -mu * s,
        mu + 2.0 * alpha,
        -mu * s,
    ], dtype=np.float64)

    roots = np.roots(coeff)
    real_nonneg = []
    for r in roots:
        if abs(r.imag) <= imag_tol and r.real >= 0.0:
            real_nonneg.append(float(r.real))

    if not real_nonneg:
        return np.empty((0,), dtype=np.float64)

    real_nonneg = np.array(sorted(real_nonneg), dtype=np.float64)
    uniq = [real_nonneg[0]]
    for val in real_nonneg[1:]:
        if abs(val - uniq[-1]) > 1e-8:
            uniq.append(val)
    return np.array(uniq, dtype=np.float64)


def _prox_logdet_sigma(s: float, mu: float, alpha: float = 1.0) -> float:
    """
    对单个奇异值 s 做 LogDet proximal：
        argmin_{x>=0} alpha*log(1+x^2) + (mu/2)*(x-s)^2

    由于该目标非凸，这里枚举所有实非负驻点，再加上边界点 0，
    用目标函数值筛选全局最优候选。
    """
    s = float(max(s, 0.0))
    mu = float(mu)
    alpha = float(alpha)

    candidates = [0.0, s]
    roots = _logdet_roots(s=s, mu=mu, alpha=alpha)
    if roots.size > 0:
        candidates.extend([float(x) for x in roots])

    best_x = 0.0
    best_val = _logdet_objective_scalar(0.0, s=s, mu=mu, alpha=alpha)
    for x in candidates:
        val = _logdet_objective_scalar(x, s=s, mu=mu, alpha=alpha)
        if val < best_val:
            best_val = val
            best_x = x
    return float(best_x)


def _logdet_shrink(mat: np.ndarray, mu: float, alpha: float = 1.0) -> np.ndarray:
    """
    矩阵版 LogDet proximal：
        prox_{(alpha/mu) * logdet(I + X^T X)}(mat)

    做法：
    1) A = U diag(s) V^T
    2) 对每个奇异值单独解标量 proximal
    3) 重构矩阵
    """
    if mu <= 0:
        raise ValueError("mu must be > 0")
    A = np.asarray(mat, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"_logdet_shrink expects a 2D matrix, got shape={A.shape}")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_new = np.empty_like(s)
    for i, si in enumerate(s):
        s_new[i] = _prox_logdet_sigma(float(si), mu=mu, alpha=alpha)
    return (U * s_new) @ Vt


# =========================
# 主方法：SSCTV-RPCA + LogDet + Elastic Net Sparse
# =========================

def ssctv_rpca_logdet(noise_data: np.ndarray, opts=None, verbose: bool = True):
    """
    SSCTV-RPCA 的 LogDet 版本。

    对应目标：
        min_{X,S}  alpha1 * logdet(I + G31^T G31)
                 + alpha2 * logdet(I + G32^T G32)
                 + 2*lambda1 * ||S||_1
                 + lambda2 * ||S||_F^2

    s.t.
        Y = X + S,
        Dx(X) = G1,
        Dy(X) = G2,
        Dz(G1) = G31,
        Dz(G2) = G32.

    参数
    ----
    noise_data : np.ndarray
        输入高光谱立方体，shape=(M,N,p)
    opts : dict
        可选参数：
        - maxIter: 最大迭代次数，默认 200
        - rho: 罚参数增长率，默认 1.03
        - tol: 收敛阈值，默认 1e-6
        - lambda1 / lambda_1: L1 稀疏权重
        - lambda2 / lambda_2: Frobenius^2 权重
        - alpha1: G31 的 LogDet 权重，默认 1.0
        - alpha2: G32 的 LogDet 权重，默认 1.0
        - mu, mu1: 初始罚参数；不传则自动初始化

    返回
    ----
    output_image : np.ndarray
        恢复结果 X，shape=(M,N,p)
    S : np.ndarray
        稀疏/残差项，shape=(M*N,p)

    备注
    ----
    - 返回接口故意保持成和原 ssctv_rpca.py 一致，便于 main.py 直接调用。
    - 若你需要在 anomaly main 里直接替换，只需：
          from methods.ssctv_rpca_logdet import ssctv_rpca_logdet
      然后像原来一样：
          X, S = ssctv_rpca_logdet(cube, opts=...)
    """
    if opts is None:
        opts = {}

    M, N, p = noise_data.shape
    sizeD = (M, N, p)

    maxIter = int(opts.get("maxIter", 200))
    rho = float(opts.get("rho", 1.03))
    tol = float(opts.get("tol", 1e-6))

    # 为了兼容原 main.py 只传一个 lambda 的写法，这里做别名兜底
    lambda1 = float(
        opts.get("lambda1",
        opts.get("lambda_1",
        opts.get("lambda1_",
        opts.get("lambda",
        opts.get("lambda_", 2.0 / np.sqrt(M * N))))))
    )
    lambda2 = float(
        opts.get("lambda2",
        opts.get("lambda_2", 0.1 * lambda1))
    )

    alpha1 = float(opts.get("alpha1", 1.0))
    alpha2 = float(opts.get("alpha2", 1.0))

    # D: (M*N, p) —— 列主序，和原实现保持一致
    D = np.reshape(noise_data.astype(np.float64), (M * N, p), order="F")
    normD = np.linalg.norm(D, ord="fro")

    # dual norm 初始化，沿用 RPCA/SSCTV 的常见做法
    smax = np.linalg.svd(D, compute_uv=False)[0]
    norm_two = smax
    norm_inf = np.linalg.norm(D.ravel(order="F"), ord=np.inf) / max(lambda1, 1e-12)
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
        # 更新 G31, G32：LogDet prox
        # -------------------------
        A1 = np.reshape(_diff_z(G1, sizeD), (M * N, p), order="F") + M4 / mu1
        G31 = _logdet_shrink(A1, mu=mu1, alpha=alpha1)

        A2 = np.reshape(_diff_z(G2, sizeD), (M * N, p), order="F") + M5 / mu1
        G32 = _logdet_shrink(A2, mu=mu1, alpha=alpha2)

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
        # 更新 S：L1 + Fro^2（Elastic Net prox）
        # min_S 2*lambda1||S||_1 + lambda2||S||_F^2 + (mu/2)||D-X-S+M1/mu||^2
        # -------------------------
        Z = D - X + M1 / mu
        S = _elastic_net_shrink(Z, lambda1=lambda1, lambda2=lambda2, mu=mu)

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
ssctv_logdet = ssctv_rpca_logdet


if __name__ == "__main__":
    # 简单自测
    np.random.seed(0)
    cube = np.random.rand(20, 18, 8)
    opts = {
        "maxIter": 5,
        "rho": 1.03,
        "tol": 1e-5,
        "lambda1": 2.0 / np.sqrt(20 * 18),
        "lambda2": 0.01,
        "alpha1": 1.0,
        "alpha2": 1.0,
    }
    X, S = ssctv_rpca_logdet(cube, opts=opts, verbose=True)
    print("X shape:", X.shape)
    print("S shape:", S.shape)
