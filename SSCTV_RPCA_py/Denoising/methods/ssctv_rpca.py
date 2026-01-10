# methods/ssctv_rpca.py
import numpy as np
from numpy.fft import fftn, ifftn


def _svd_shrink(mat: np.ndarray, tau: float) -> np.ndarray:
    """核范数近端：对奇异值作软阈值"""
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    s_thr = np.maximum(s - tau, 0.0)
    return (U * s_thr) @ Vt


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """元素级软阈值"""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


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
    d = -(cube - np.roll(cube, 1, axis=0))   # 负的后向差分
    return np.reshape(d, (M * N, p), order="F")

def _diff_yT(Y_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(Y_vec, (M, N, p), order="F")
    d = -(cube - np.roll(cube, 1, axis=1))   # 负的后向差分
    return np.reshape(d, (M * N, p), order="F")

def _diff_zT(Y_vec, sizeD):
    M, N, p = sizeD
    cube = np.reshape(Y_vec, (M, N, p), order="F")
    d = -(cube - np.roll(cube, 1, axis=2))   # 负的后向差分
    return np.reshape(d, (M * N, p), order="F")



def _freq_energy(n: int) -> np.ndarray:
    """|FFT([1,-1])|^2 的采样（与 psf2otf([1,-1]) 的模方等价）"""
    k = np.arange(n, dtype=np.float64)
    return 2.0 - 2.0 * np.cos(2.0 * np.pi * k / n)


def ssctv_rpca(noise_data: np.ndarray, opts=None, verbose=True):
    """
    Python 版 SSCTV-RPCA（与原 m 文件等价）
    输入:
        noise_data: (M,N,p) numpy 数组
        opts: dict，可给 {maxIter,rho,tol,lambda 或 lambda_}
    返回:
        output_image: (M,N,p) 去噪结果
        E: (M*N,p) 稀疏噪声矩阵（与原实现一致）
    """
    if opts is None:
        opts = {}
    M, N, p = noise_data.shape
    maxIter = int(opts.get("maxIter", 200))
    rho = float(opts.get("rho", 1.03))
    tol = float(opts.get("tol", 1e-6))
    lambda_ = float(opts.get("lambda", opts.get("lambda_", 2.0 / np.sqrt(M * N))))

    sizeD = (M, N, p)

    # D: (M*N, p)  —— 注意列主序
    D = np.reshape(noise_data.astype(np.float64), (M * N, p), order="F")
    normD = np.linalg.norm(D, ord="fro")

    # 近似最大奇异值（谱范数）
    # 为稳妥起见取完整 SVD 的首个奇异值；如需更快可换 power iteration
    smax = np.linalg.svd(D, compute_uv=False)[0]
    norm_two = smax
    norm_inf = np.linalg.norm(D.ravel(order="F"), ord=np.inf) / max(lambda_, 1e-12)
    dual_norm = max(norm_two, norm_inf)

    mu = 1.0 / max(dual_norm, 1e-12)
    mu1 = 1.0 * mu
    max_mu = mu * 1e7

    # 频域能量项（等价 Eny_x/Eny_y/Eny_z）
    Enx = _freq_energy(M)[:, None, None]
    Eny = _freq_energy(N)[None, :, None]
    Enz = _freq_energy(p)[None, None, :]

    determ_xy = Enx + Eny               # Dx^T Dx + Dy^T Dy
    determ_z = Enz                      # Dz^T Dz

    # 初始化变量
    X = D.copy()
    G1 = np.zeros_like(D)
    G2 = np.zeros_like(D)
    E = np.zeros_like(D)

    M1 = D / max(dual_norm, 1e-12)
    M2 = M1.copy()
    M3 = M1.copy()
    M4 = M1.copy()
    M5 = M1.copy()

    for it in range(1, maxIter + 1):
        # --- 更新 G31, G32  （对 Dz(G1)、Dz(G2) 做核范数近端）
        A = np.reshape(_diff_z(G1, sizeD), (M * N, p), order="F") + M4 / mu1
        G31 = _svd_shrink(A, 1.0 / mu1)

        A = np.reshape(_diff_z(G2, sizeD), (M * N, p), order="F") + M5 / mu1
        G32 = _svd_shrink(A, 1.0 / mu1)

        # --- 更新 G1
        diffT_p = _diff_zT(mu1 * G31 - M4, sizeD)  # (M*N,p)
        numer = diffT_p + mu1 * _diff_x(X, sizeD) + M2      # (M*N,p)
        num_cube = np.reshape(numer, (M, N, p), order="F")
        den = (mu1 * determ_z + mu1)        # (M,N,p) + broadcast
        x = np.real(ifftn(fftn(num_cube) / den))
        G1 = np.reshape(x, (M * N, p), order="F")

        # --- 更新 G2
        diffT_p = _diff_zT(mu1 * G32 - M5, sizeD)
        numer = diffT_p + mu1 * _diff_y(X, sizeD) + M3
        num_cube = np.reshape(numer, (M, N, p), order="F")
        den = (mu1 * determ_z + mu1)
        x = np.real(ifftn(fftn(num_cube) / den))
        G2 = np.reshape(x, (M * N, p), order="F")

        # --- 更新 X
        diffT_p = _diff_xT(mu1 * G1 - M2, sizeD) + _diff_yT(mu1 * G2 - M3, sizeD)
        numer = diffT_p + mu * (D - E) + M1
        num_cube = np.reshape(numer, (M, N, p), order="F")
        den = (mu1 * determ_xy + mu)
        x = np.real(ifftn(fftn(num_cube) / den))
        X = np.reshape(x, (M * N, p), order="F")

        # --- 更新 E（L1 近端）
        E = _soft_threshold(D - X + M1 / mu, lambda_ / mu)

        # --- 计算残差并判停
        leq1 = D - X - E
        leq2 = np.reshape(_diff_x(X, sizeD), (M * N, p), order="F") - G1
        leq3 = np.reshape(_diff_y(X, sizeD), (M * N, p), order="F") - G2
        leq4 = np.reshape(_diff_z(G1, sizeD), (M * N, p), order="F") - G31
        leq5 = np.reshape(_diff_z(G2, sizeD), (M * N, p), order="F") - G32

        stopC1 = np.linalg.norm(leq1, ord="fro") / (normD + 1e-12)
        stopC2 = np.max(np.abs(leq2))
        stopC4 = np.linalg.norm(leq4, ord="fro") / (normD + 1e-12)

        if verbose and (it % 10 == 0 or it == 1):
            print(f"iter {it:3d}, mu={mu:8.2e}, Y-X-E={stopC1:8.2e}, "
                  f"||DX-G1||_inf={stopC2:8.2e}, |Dz(G1)-G31|_F/normD={stopC4:8.2e}")

        if (stopC1 < tol) and (stopC2 < tol):
            break

        # --- 拉格朗日乘子和惩罚参数
        M1 = M1 + mu * leq1
        M2 = M2 + mu1 * leq2
        M3 = M3 + mu1 * leq3
        M4 = M4 + mu1 * leq4
        M5 = M5 + mu1 * leq5

        mu = min(max_mu, mu * rho)
        mu1 = min(max_mu, mu1 * rho)

    output_image = np.reshape(X, (M, N, p), order="F")
    return output_image, E
