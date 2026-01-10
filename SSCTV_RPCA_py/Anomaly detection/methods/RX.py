# rx.py
# -*- coding: utf-8 -*-
import numpy as np

def RX(X, reg: float = 1e-8):
    """
    Python 等价版 RX 检测器（Mahalanobis 距离）
    Parameters
    ----------
    X : ndarray, shape (N, M)
        每列一个样本
    reg : float
        协方差正则化强度（加到对角上），避免奇异

    Returns
    -------
    D : ndarray, shape (M,)
        每列样本的 RX 分数
    """
    X = np.asarray(X, dtype=np.float64)
    N, M = X.shape
    if M < 2:
        return np.zeros(M, dtype=np.float64)

    mu = X.mean(axis=1, keepdims=True)     # (N,1)
    Xc = X - mu                             # (N,M)

    # 协方差： (Xc Xc^T)/(M-1)
    Sigma = (Xc @ Xc.T) / (M - 1)
    # 正则化（与 inv 对齐更稳定）
    Sigma = Sigma + reg * np.eye(N, dtype=np.float64)

    try:
        invS = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(Sigma)

    # D_m = x_m^T invS x_m  （向量化）
    Y = invS @ Xc                         # (N,M)
    D = np.sum(Xc * Y, axis=0)           # (M,)
    # 数值保护：微小负数置零
    D[D < 0] = 0.0
    return D


# quick test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, M = 10, 100
    X = rng.standard_normal((N, M))
    D = RX(X)
    print(D.shape, D.min(), D.max())
