# ad_lilu7.py
# -*- coding: utf-8 -*-
import numpy as np

def AD_lilu7(Data: np.ndarray, lambda_: float):
    """
    Python 等价版 AD_lilu7：
      - 先在 (H*W, Dim) 上做 SVD，依据 (||Y||_F * lambda)^2 与奇异值平方比较得到截断起始下标 np_idx；
      - 对每个像素的 3x3 周期邻域构成张量 (3,3,Dim,num)，去均值后对谱维做 FFT；
      - 计算每个邻域位置 (3x3 共 9 个) 的谱协方差矩阵 G，并取其特征向量（原版用 SVD）；
      - 仅保留从 np_idx 以后（对应“较高频/能量较小”部分）的 FFT 系数，通过 U[:, np_idx:] 映射回 Dim 维，
        再做 IFFT 得到时域；
      - 对每个样本、每个波段，把 3x3 的值求和，得到 Anomaly(num, Dim)。

    Parameters
    ----------
    Data : ndarray, shape (H, W, Dim)
    lambda_ : float

    Returns
    -------
    Anomaly : ndarray, shape (H*W, Dim)
    """
    H, W, Dim = Data.shape
    num = H * W

    # ---------- 阶段1：在 (H*W, Dim) 上 SVD，取阈值 ----------
    Y = Data.reshape(H * W, Dim, order='F')  # 与 MATLAB 列主序一致
    # Frobenius 范数 = sqrt(奇异值平方和)
    # 仅需奇异值，不需要 U/V
    S = np.linalg.svd(Y, full_matrices=False, compute_uv=False)
    E_tol = np.sqrt(np.sum(S**2))
    thre = (E_tol * lambda_) ** 2

    # MATLAB: np = find(S.^2 < thre); np = np(1);
    idxs = np.where(S**2 < thre)[0]
    # 若找不到，原 MATLAB 会报错；这里做个更稳的退化：从 Dim 开始（等于空保留）
    np_idx = int(idxs[0]) if idxs.size > 0 else Dim  # 0-based

    # ---------- 阶段2：3×3 周期邻域张量 ----------
    # 周期平移（roll）构造 9 个邻居（向上/下/左/右及四个对角）
    up    = np.roll(Data,  1, axis=0)  # (i-1)%H
    down  = np.roll(Data, -1, axis=0)  # (i+1)%H
    left  = np.roll(Data,  1, axis=1)  # (j-1)%W
    right = np.roll(Data, -1, axis=1)  # (j+1)%W

    up_left    = np.roll(up,    1, axis=1)
    up_right   = np.roll(up,   -1, axis=1)
    down_left  = np.roll(down,  1, axis=1)
    down_right = np.roll(down, -1, axis=1)

    # 把每个邻域位置整理成 (Dim, num) 并放入 Ten[3,3,Dim,num]
    def to_Dim_num(arr):
        # arr: (H,W,Dim) -> (Dim, H*W) with Fortran order to match MATLAB的 j外i内 顺序
        return arr.transpose(2, 0, 1).reshape(Dim, num, order='F')

    Ten = np.empty((3, 3, Dim, num), dtype=np.complex128)
    Ten[0, 0] = to_Dim_num(up_left)
    Ten[0, 1] = to_Dim_num(up)
    Ten[0, 2] = to_Dim_num(up_right)
    Ten[1, 0] = to_Dim_num(left)
    Ten[1, 1] = to_Dim_num(Data)
    Ten[1, 2] = to_Dim_num(right)
    Ten[2, 0] = to_Dim_num(down_left)
    Ten[2, 1] = to_Dim_num(down)
    Ten[2, 2] = to_Dim_num(down_right)

    # 去均值（沿样本维 num）
    X_m = Ten.mean(axis=3, keepdims=False)              # (3,3,Dim)
    Ten_centered = Ten - X_m[:, :, :, None]             # (3,3,Dim,num)

    # 谱维 FFT（沿 Dim 轴=2）
    Ten_fft = np.fft.fft(Ten_centered, axis=2)          # (3,3,Dim,num)

    # 仅保留从 np_idx 开始的频率分量（MATLAB 的 np:end）
    Ten_hat_fft = Ten_fft[:, :, np_idx:, :]             # (3,3,Dim-np_idx,num)

    # ---------- 阶段3：对每个邻域位置做协方差 + SVD，并重建 ----------
    Ten_hat_ifft = np.zeros((3, 3, Dim, num), dtype=np.complex128)

    # 9 个位置，逐个处理
    for i in range(3):
        for j in range(3):
            # 用“完整频谱”的 FFT 来估计协方差（与原代码一致：用 Ten_fft 而非裁剪后的 Ten_hat_fft）
            Fij = Ten_fft[i, j]               # (Dim, num)
            # 协方差估计 G = (1/(num-1)) * sum_k f_k f_k^H = Fij @ Fij^H / (num-1)
            G = (Fij @ Fij.conj().T) / max(1, num - 1)  # (Dim, Dim), Hermitian PSD

            # SVD（原代码用 svd(G)），取 U 的列
            # 对 Hermitian 也可以用 eigh，这里保持与原版一致
            U, Sg, Vh = np.linalg.svd(G, full_matrices=False)

            # 取 U 的列从 np_idx 开始（MATLAB 的 u(:,np:end)）
            if np_idx < Dim:
                U_eps = U[:, np_idx:]                    # (Dim, r)
                # 对应的被保留频率分量（与 U_eps 列数一致）
                Fhat = Ten_hat_fft[i, j]                 # (Dim-np_idx, num)
                # 回映射到 Dim
                rec = U_eps @ Fhat                       # (Dim, num)
            else:
                # 无保留分量 -> 直接全零
                rec = np.zeros((Dim, num), dtype=np.complex128)

            # 逆 FFT（沿谱维）
            Ten_hat_ifft[i, j] = np.fft.ifft(rec, axis=0)

    # ---------- 阶段4：聚合为 Anomaly(num, Dim) ----------
    # 对每个样本、每个波段，把 3×3 的值求和
    # Ten_hat_ifft: (3,3,Dim,num) -> sum over (0,1) -> (Dim,num) -> 转置 (num,Dim)
    Anomaly = Ten_hat_ifft.sum(axis=(0, 1)).T           # (num, Dim)
    # 原始 MATLAB 没有显式取实部，但数据通常实；这里取接近实数的实部
    Anomaly = np.real_if_close(Anomaly, tol=1e8)
    return Anomaly
