# godec.py
# -*- coding: utf-8 -*-
import numpy as np

def GoDec(X, rank, card, power=0, iter_max=100, error_bound=1e-3, random_state=None):
    """
    Python 等价版 GoDec (Zhou & Tao, ICML 2011)

    min  ||X - L - S||_F
    s.t. rank(L) <= rank,  |S|_0 <= card

    Parameters
    ----------
    X : ndarray, shape (m, n)
        数据矩阵（样本×特征）
    rank : int
        低秩部分的秩上界
    card : int
        稀疏部分非零元素个数上界
    power : int, optional (default=0)
        幂迭代次数，越大越准但越慢（原脚本：power>=0）
    iter_max : int, optional (default=100)
        最大迭代次数（原脚本 1e2）
    error_bound : float, optional (default=1e-3)
        残差阈值（与原脚本一致）
    random_state : int or None
        随机种子（控制 randn 初始化）

    Returns
    -------
    L : ndarray
        低秩部分，shape 与输入 X 相同
    S : ndarray
        稀疏部分，shape 与输入 X 相同
    RMSE : ndarray, shape (t,)
        每次迭代的残差向量范数（与原脚本一致：norm(T(:)))
    error : float
        相对误差 ||X - L - S||_F / ||X||_F
    """
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape

    # 与 MATLAB 对齐：若 m<n 则内部转置，最后转回
    transposed = False
    if m < n:
        X = X.T
        m, n = X.shape
        transposed = True

    # 初始化
    L = X.copy()
    S = np.zeros_like(X)  # MATLAB 初始化为 sparse(zeros(size(X)))，这里用 dense 即可
    RMSE = []

    # 随机数
    rng = np.random.default_rng(random_state)

    # clip rank 与 card
    r_eff = int(max(1, min(rank, min(m, n))))
    k_eff = int(max(0, min(card, X.size)))

    it = 0
    while True:
        it += 1

        # ---- Update L：随机子空间 + 幂迭代 + QR 投影 ----
        Y2 = rng.standard_normal((n, r_eff))
        for _ in range(power + 1):
            Y1 = L @ Y2      # (m, r)
            Y2 = L.T @ Y1    # (n, r)
        Q, _ = np.linalg.qr(Y2, mode='reduced')  # (n, r)
        L_new = (L @ Q) @ Q.T                    # (m, n)

        # ---- Update S：保留 |T| 最大的 card 个元素 ----
        T = L - L_new + S
        L = L_new
        if k_eff > 0:
            # 先用 argpartition 找到 top-k，再在这些里排序，等价且更快
            flat = np.abs(T).ravel()
            if k_eff < flat.size:
                idx_part = np.argpartition(flat, -k_eff)[-k_eff:]
                # 精确排序（降序）
                idx_sorted = idx_part[np.argsort(flat[idx_part])[::-1]]
            else:
                idx_sorted = np.argsort(flat)[::-1]  # 全部
            S = np.zeros_like(X)
            S.reshape(-1)[idx_sorted] = T.reshape(-1)[idx_sorted]
            # 将稀疏成分从 T 中剔除，用于 RMSE 与后续 L 更新
            T.reshape(-1)[idx_sorted] = 0.0
        else:
            S.fill(0.0)

        # ---- 误差与停止条件 ----
        RMSE.append(np.linalg.norm(T.ravel()))
        if (RMSE[-1] < error_bound) or (it >= iter_max):
            break
        else:
            L = L + T  # 与原脚本一致

    LS = L + S
    rel_err = np.linalg.norm(LS - X) / (np.linalg.norm(X) + 1e-16)

    if transposed:
        L = L.T
        S = S.T
        # LS 只用于计算 error，无需返回；若你需要可一并返回
    return L, S, np.asarray(RMSE), rel_err


# ---- quick test ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, n = 200, 50
    # 合成低秩 + 稀疏 + 噪声
    U = rng.standard_normal((m, 5))
    V = rng.standard_normal((n, 5))
    L0 = U @ V.T
    S0 = np.zeros((m, n))
    idx = rng.choice(m*n, size=500, replace=False)
    S0.reshape(-1)[idx] = rng.standard_normal(500) * 5
    X = L0 + S0 + 0.1 * rng.standard_normal((m, n))

    L, S, RMSE, err = GoDec(X, rank=5, card=500, power=1, iter_max=100, error_bound=1e-3, random_state=0)
    print("Shapes:", L.shape, S.shape, "| RMSE last:", RMSE[-1], "| rel_err:", err)
