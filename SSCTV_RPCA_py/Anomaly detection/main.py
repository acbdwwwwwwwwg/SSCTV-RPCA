# -*- coding: utf-8 -*-
"""
CPU runner for Hyperspectral Anomaly Detection (fixed).

修正要点：
1) 归一化改为“max–min 到 [0,1]”（与论文一致），替换原先的逐像素 L2 归一化。
2) GT 与数据立方体采用“中心对齐”法，避免 Urban 等数据集因尺寸不一致而错位。
3) 其他逻辑保持原样：并行、结果落盘、ROC/AUC 计算、可视化等。

输出增强：
A) 运行时：显示“第几个数据集/总数、数据集名、方法名、效果（AUC 或 N/A）”，以及任务进度。
B) 运行结束：打印所有数据集的汇总表（长表 + AUC 透视表），并保存 CSV/JSON。

你这次要的改动（加入 CAVE 多光谱数据集）：
✅ 自动扫描 ./data/CAVE 下的 32 个 case（每个 case 目录一般以 *_ms 结尾）
✅ 每个 case 读取 31 张 band 图（*_ms_01.png ~ *_ms_31.png）叠成 (H,W,B) 立方体
✅ RGB bmp 会自动忽略
✅ 默认无 GT：会跳过 ROC/AUC，但仍会输出 score/heatmap 和统计量

运行方式：
- PyCharm 直接 Run：默认 --only=all，跑 ds_all 声明顺序的全部数据集（含 CAVE）
- 想只跑 CAVE：--only=CAVE_（见下方 only 的“前缀匹配”逻辑）
"""

import argparse, os, sys, json, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from time import perf_counter
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- threading hygiene --------------------------------
# 防止 BLAS 过度并行导致进程过载
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------ paths -----------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METHODS_DIR = ROOT / "methods"
RESULT_DIR = ROOT / "result"

if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

# Optional imports (guarded)
try:
    from methods.ssctv_rpca import ssctv_rpca   # expects: X,E = ssctv_rpca(cube, opts=dict(...))
except Exception:
    ssctv_rpca = None
try:
    from methods.Unsupervised_RPCA_Detect_v1 import Unsupervised_RPCA_Detect_v1
except Exception:
    Unsupervised_RPCA_Detect_v1 = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------ I/O helpers -----------------------------------

def load_cube_from_mat(path: Path) -> np.ndarray:
    """从 .mat 加载 3D 立方体。按常见 key 顺序猜测变量名。"""
    md = loadmat(str(path))
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim == 3]
    if not candidates:
        raise RuntimeError(f"No 3D cube found in {path.name}. Keys: {list(md.keys())[:10]}")
    order = ['Urban', 'Sandiego_new', 'Sandiego', 'PaviaU', 'HSI', 'Y', 'data', 'cube', 'M', 'X']
    for name in order:
        for k, v in candidates:
            if name.lower() in k.lower():
                return v.astype(np.float64)
    _, v = max(candidates, key=lambda kv: kv[1].size)
    return v.astype(np.float64)

def load_gt_from_mat(path: Path) -> np.ndarray:
    """从 .mat 加载 GT（1/0）。尽量容忍不同命名与形状。"""
    md = loadmat(str(path))
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim in (1, 2)]
    if not candidates:
        raise RuntimeError(f"No GT-like array found in {path.name}. Keys: {list(md.keys())[:10]}")
    order = ['UGt', 'gt', 'GT', 'Sandiego_gt', 'GroundTruth', 'truth', 'mask']
    for name in order:
        for k, v in candidates:
            if name.lower() in k.lower():
                return v.astype(np.float64)
    _, v = max(candidates, key=lambda kv: kv[1].size)
    return v.astype(np.float64)

def _imread_any(path: Path) -> np.ndarray:
    """
    读取 png/bmp/jpg 等图像为 numpy。
    - 若读到 3 通道：默认取灰度（取第 1 通道；CAVE 的 band png 通常本来就是单通道）
    """
    try:
        import imageio.v2 as imageio  # type: ignore
        img = imageio.imread(str(path))
    except Exception:
        from PIL import Image  # type: ignore
        img = np.array(Image.open(path))

    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float64)

def load_cube_from_cave_case(case_dir: Path, expected_bands: int = 31) -> np.ndarray:
    """
    读取 CAVE 一个 case：
    - case_dir: 指向包含 *_ms_01.png ... *_ms_31.png 的目录
    - 自动按数字后缀排序叠成 (H,W,B)
    """
    if not case_dir.exists():
        raise RuntimeError(f"CAVE case dir not found: {case_dir}")

    # 找 png：优先匹配 *_ms_XX.png
    pngs = list(case_dir.glob("*_ms_*.png"))
    if not pngs:
        # 再宽松一点
        pngs = list(case_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG bands found under: {case_dir}")

    def _band_index(p: Path) -> int:
        # 匹配末尾 _01 / _31 之类
        m = re.search(r"_([0-9]{1,3})\.png$", p.name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        # 若不匹配，给个很大的序号，保证排序稳定
        return 10**9

    pngs_sorted = sorted(pngs, key=lambda p: (_band_index(p), p.name.lower()))

    # 若里面混入了非 band png（极少见），我们尽量只保留有数字后缀的
    has_index = [p for p in pngs_sorted if _band_index(p) < 10**9]
    if len(has_index) >= 2:
        pngs_sorted = has_index

    if expected_bands is not None and len(pngs_sorted) != expected_bands:
        print(f"[WARN] CAVE case {case_dir} has {len(pngs_sorted)} band PNGs (expected {expected_bands}). Will load all found.", flush=True)

    bands = []
    for p in pngs_sorted:
        bands.append(_imread_any(p))
    # shape check
    h, w = bands[0].shape
    for i, b in enumerate(bands, 1):
        if b.shape != (h, w):
            raise RuntimeError(f"Inconsistent band shape in {case_dir}: band#{i} {b.shape} vs {(h,w)}")

    cube = np.stack(bands, axis=2)  # H,W,B
    return cube.astype(np.float64)

def _find_cave_inner_dir(case_root: Path) -> Path:
    """
    适配你给的结构：data/CAVE/balloons_ms/balloons_ms/...
    - 优先用 case_root / case_root.name
    - 否则用 case_root 本身
    """
    candidate = case_root / case_root.name
    if candidate.exists() and candidate.is_dir():
        return candidate
    return case_root

def discover_cave_datasets(cave_root: Path) -> Dict[str, "Dataset"]:
    """
    扫描 ./data/CAVE 下所有 case，返回一个 dict 可直接 update 到 ds_all：
    key: CAVE_<case_folder_name>
    Dataset.name 同 key，cube_path 指向包含 band png 的目录，kind='cave'
    """
    out: Dict[str, Dataset] = {}
    if not cave_root.exists() or not cave_root.is_dir():
        return out

    for case_root in sorted([p for p in cave_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        inner = _find_cave_inner_dir(case_root)

        # 基本判定：至少有 *_ms_*.png 或 .png
        pngs = list(inner.glob("*_ms_*.png"))
        if not pngs:
            pngs = list(inner.glob("*.png"))
        if not pngs:
            continue

        key = f"CAVE_{case_root.name}"
        out[key] = Dataset(name=key, cube_path=inner, gt_path=None, kind="cave")

    return out

# ------------------------------- normalization --------------------------------

def minmax_normalize_cube(cube: np.ndarray) -> np.ndarray:
    """论文口径：max–min 归一化到 [0,1]（全局线性缩放）。"""
    vmin = float(np.nanmin(cube))
    vmax = float(np.nanmax(cube))
    denom = max(vmax - vmin, 1e-12)
    out = (cube - vmin) / denom
    out[np.isnan(out)] = 0.0
    return out

# ----------------------------- visualization ----------------------------------

def save_heatmap(score: np.ndarray, out_png: Path, cmap="viridis"):
    plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(score, cmap=cmap)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
    plt.close()

# ------------------------------- ROC / AUC ------------------------------------

def roc_curve_pf_pd(scores: np.ndarray, gt: np.ndarray):
    """按分数递减阈值计算 PF/PD（FPR/TPR）。"""
    s = scores.reshape(-1).astype(float)
    y = gt.reshape(-1).astype(int)
    y = (y > 0).astype(int)
    order = np.argsort(-s)
    y_sorted = y[order]
    P = np.sum(y_sorted == 1)
    N = np.sum(y_sorted == 0)
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y_sorted == 1).astype(float)
    fp = np.cumsum(y_sorted == 0).astype(float)
    pd = tp / P
    pf = fp / N
    pf = np.concatenate(([0.0], pf, [1.0]))
    pd = np.concatenate(([0.0], pd, [1.0]))
    return pf, pd

def auc_trapz(pf: np.ndarray, pd: np.ndarray) -> float:
    order = np.argsort(pf)
    return float(np.trapz(pd[order], pf[order]))

# ---------------------- residual ↔ score / cube adapters ----------------------

def _score_from_residual(residual: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """把残差/稀疏项 E/S 转成 (H,W) 的分数图，自动适配常见形状。"""
    H, W, B = cube.shape
    R = residual
    if R is None:
        raise RuntimeError("Residual/Sparse component is None.")

    # 3D：凑成 (H,W,B) 后按谱向取 ℓ2
    if R.ndim == 3:
        if R.shape == (H, W, B):
            return np.linalg.norm(R, axis=2)
        for axes in ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)):
            Rt = np.transpose(R, axes)
            if Rt.shape == (H, W, B):
                return np.linalg.norm(Rt, axis=2)

    # 2D：常见的 (H*W,B) / (B,H*W) / 直接 (H,W)
    if R.ndim == 2:
        if R.shape == (H, W):           # 已是分数图
            return R
        if R.shape == (W, H):           # 转置一下
            return R.T
        if R.shape[1] == B:             # (H*W, B)
            return np.linalg.norm(R, axis=1).reshape(H, W, order='F')
        if R.shape[0] == B:             # (B, H*W)
            return np.linalg.norm(R, axis=0).reshape(H, W, order='F')
        if R.size == H * W * B:         # 2D 扁平化了 3D
            Rt = R.reshape(H, W, B, order='F')
            return np.linalg.norm(Rt, axis=2)
        if R.size == H * W:             # 扁平分数
            return R.reshape(H, W, order='F')

    # 1D：扁平分数
    if R.ndim == 1 and R.size == H * W:
        return R.reshape(H, W, order='F')

    raise RuntimeError(f"Cannot infer score map from residual shape {R.shape} for cube {(H,W,B)}.")

def _cube_from_residual(residual: np.ndarray, cube: np.ndarray) -> Optional[np.ndarray]:
    """尽量把残差/稀疏项转为 (H,W,B) 立方体；失败则返回 None。"""
    H, W, B = cube.shape
    R = residual
    if R is None:
        return None
    if R.ndim == 3:
        if R.shape == (H, W, B):
            return R
        for axes in ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)):
            Rt = np.transpose(R, axes)
            if Rt.shape == (H, W, B):
                return Rt
        return None
    if R.ndim == 2:
        if R.shape == (H * W, B):
            return R.reshape(H, W, B, order='F')
        if R.shape == (B, H * W):
            return R.reshape(B, H, W, order='F').transpose(1, 2, 0)
        if R.size == H * W * B:
            return R.reshape(H, W, B, order='F')
        return None
    return None

# ------------------------------ detectors -------------------------------------

def detector_RX(cube: np.ndarray) -> np.ndarray:
    H, W, B = cube.shape
    X = cube.reshape(-1, B).T   # B x N
    M = X.shape[1]
    mu = np.mean(X, axis=1, keepdims=True)
    Xc = X - mu
    Sigma = (Xc @ Xc.T) / max(M - 1, 1)
    Sigma += 1e-6 * np.eye(B)
    Sigma_inv = np.linalg.inv(Sigma)
    AX = Sigma_inv @ Xc   # B x N
    d = np.sum(Xc * AX, axis=0)  # length-N, Mahalanobis squared distance
    return d.reshape(H, W)

def detector_RPCA_RX(cube: np.ndarray, lam: float) -> np.ndarray:
    """先做 RPCA 得到 S，再在 S 的谱上跑 RX。若无法还原成立方体，则退化为分数图。"""
    if Unsupervised_RPCA_Detect_v1 is None:
        raise RuntimeError("Unsupervised_RPCA_Detect_v1.py not found in methods/.")
    try:
        _, S, _ = Unsupervised_RPCA_Detect_v1(cube, lam)
    except TypeError:
        out = Unsupervised_RPCA_Detect_v1(cube, lam)
        if isinstance(out, (list, tuple)):
            S = out[1] if len(out) >= 2 else out[0]
        else:
            raise
    S_cube = _cube_from_residual(S, cube)
    if S_cube is not None:
        return detector_RX(S_cube)
    return _score_from_residual(S, cube)

def detector_SSCTV(cube: np.ndarray, lam: float, opts: Dict = None) -> np.ndarray:
    if ssctv_rpca is None:
        raise RuntimeError("ssctv_rpca.py not found in methods/.")
    local_opts = dict(lambda_=lam, lambdaVal=lam, lambda_val=lam, maxIter=1000, rho=1.03, tol=1e-6)
    if opts:
        local_opts.update(opts)
    try:
        _, E = ssctv_rpca(cube, opts=local_opts)
    except TypeError:
        _, E = ssctv_rpca(cube, **local_opts)
    return _score_from_residual(E, cube)

# ---------------------------- dataset handling --------------------------------

@dataclass
class Dataset:
    name: str
    cube_path: Path                 # mat: *.mat；cave: case 目录（包含 *_ms_01.png ...）
    gt_path: Optional[Path] = None
    kind: str = "mat"               # "mat" or "cave"

def center_align_gt(cube: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """将 GT 与数据立方体做“中心对齐”：把两者裁成相同尺寸并居中贴合。"""
    H, W, _ = cube.shape

    # 1) 1D 情况：恰好 H*W，直接还原为图像
    if gt.ndim == 1 and gt.size == H * W:
        return (gt.reshape(H, W, order='F') > 0).astype(np.uint8)

    # 2) 已同尺寸
    if gt.ndim == 2 and (gt.shape[0] == H and gt.shape[1] == W):
        return (gt > 0).astype(np.uint8)

    # 3) 中心对齐裁剪
    out = np.zeros((H, W), dtype=np.uint8)
    h = min(H, gt.shape[0])
    w = min(W, gt.shape[1])

    H0 = (H - h) // 2
    W0 = (W - w) // 2
    G0 = (gt.shape[0] - h) // 2
    G1 = (gt.shape[1] - w) // 2

    out[H0:H0 + h, W0:W0 + w] = (gt[G0:G0 + h, G1:G1 + w] > 0).astype(np.uint8)
    return out

# ------------------------------ pretty output ---------------------------------

def _fmt(x, nd=4):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    """简单 ASCII 表格（不依赖第三方库）。"""
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))

    def line(ch="-"):
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def row(vals):
        return "|" + "|".join(f" {str(vals[i]).ljust(widths[i])} " for i in range(cols)) + "|"

    out = [line("-"), row(headers), line("=")]
    for r in rows:
        out.append(row(r))
        out.append(line("-"))
    return "\n".join(out)

# ------------------------------ task logic ------------------------------------

def _run_one_method_task(
    dataset: str,
    cube_kind: str,
    cube_path: str,
    gt_path: Optional[str],
    method: str,
    out_dir: str,
    ssctv_opts: Dict,
    ds_idx: int,
    ds_total: int,
    task_idx: int,
    task_total: int,
) -> Dict:
    """A single (dataset, method) unit of work. Executed in a separate process."""
    # Re-import methods in subprocess context
    if str(METHODS_DIR) not in sys.path:
        sys.path.insert(0, str(METHODS_DIR))
    from pathlib import Path as _P
    import numpy as _np

    # lightweight reimports (ensure detector sees correct globals)
    try:
        from methods.ssctv_rpca import ssctv_rpca as _ssctv_rpca
    except Exception:
        _ssctv_rpca = None
    try:
        from methods.Unsupervised_RPCA_Detect_v1 import Unsupervised_RPCA_Detect_v1 as _URD
    except Exception:
        _URD = None

    # --- make them visible to detector_* (override module globals) ---
    global ssctv_rpca, Unsupervised_RPCA_Detect_v1
    ssctv_rpca = _ssctv_rpca
    Unsupervised_RPCA_Detect_v1 = _URD

    t0 = perf_counter()
    prefix = f"[DS {ds_idx}/{ds_total} {dataset} | TASK {task_idx}/{task_total} | {method}]"
    print(f"{prefix} START", flush=True)

    # IO: cube
    cube_k = (cube_kind or "mat").lower().strip()
    cube_p = _P(cube_path)

    if cube_k == "mat":
        cube = load_cube_from_mat(cube_p)
    elif cube_k == "cave":
        cube = load_cube_from_cave_case(cube_p, expected_bands=31)
    else:
        raise RuntimeError(f"Unknown cube_kind='{cube_kind}' for dataset '{dataset}'")

    # IO: GT (optional)
    gt0 = None
    if gt_path:
        try:
            gt_p = _P(gt_path)
            if gt_p.exists():
                gt0 = load_gt_from_mat(gt_p)
            else:
                print(f"{prefix} [WARN] GT file not found: {gt_path}. Skip ROC/AUC.", flush=True)
        except Exception as e:
            print(f"{prefix} [WARN] failed to load GT from {gt_path}: {e}. Skip ROC/AUC.", flush=True)

    # 论文口径归一化
    cube = minmax_normalize_cube(cube)

    # GT（可选）：若提供则与数据立方体做中心对齐
    GT = center_align_gt(cube, gt0) if gt0 is not None else None
    has_gt = GT is not None

    H, W, _B = cube.shape
    lam = 1.0 / _np.sqrt(H * W)

    # run method
    m = method.upper()
    if m == "RX":
        score = detector_RX(cube)
    elif m in ("RPCA_RX", "RPCA-RX"):
        if Unsupervised_RPCA_Detect_v1 is None:
            raise RuntimeError("Unsupervised_RPCA_Detect_v1.py not found in methods/.")
        score = detector_RPCA_RX(cube, lam)
    elif m in ("SSCTV", "SSCTV_RPCA", "SSCTV-RPCA"):
        if ssctv_rpca is None:
            raise RuntimeError("ssctv_rpca.py not found in methods/.")
        score = detector_SSCTV(cube, lam, ssctv_opts)
    else:
        raise RuntimeError(f"Unknown method '{method}'")

    # stats（无 GT 时也能给你一个“效果感知”）
    score = score.astype(np.float64, copy=False)
    score_min = float(np.nanmin(score))
    score_max = float(np.nanmax(score))
    score_mean = float(np.nanmean(score))
    score_std = float(np.nanstd(score))

    # normalize for viz
    smin, smax = score_min, score_max
    score_vis = (score - smin) / (smax - smin) if smax > smin else np.zeros_like(score)

    # Save artifacts (score/heatmap always)
    out_dir_p = _P(out_dir)
    ensure_dir(out_dir_p)
    base = f"{dataset}_{method}"
    _np.save(out_dir_p / f"{base}_score.npy", score.astype(np.float32))
    save_heatmap(score_vis, out_dir_p / f"{base}_score.png")

    auc = None
    if has_gt:
        pf, pd = roc_curve_pf_pd(score, GT)
        auc = auc_trapz(pf, pd)

        plt.figure(figsize=(4, 4), dpi=160)
        plt.plot(pf, pd, lw=2)
        plt.xlabel("PF (False Positive Rate)")
        plt.ylabel("PD (True Positive Rate)")
        plt.title(f"ROC • {dataset} • {method} • AUC={auc:.4f}")
        plt.grid(True, ls="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_dir_p / f"{base}_ROC.png")
        plt.close()

    seconds = float(perf_counter() - t0)
    auc_str = "N/A" if auc is None else f"{auc:.4f}"
    print(f"{prefix} DONE | AUC={auc_str} | time={seconds:.2f}s", flush=True)

    return {
        "ds_idx": ds_idx,
        "ds_total": ds_total,
        "task_idx": task_idx,
        "task_total": task_total,
        "dataset": dataset,
        "method": method,
        "has_gt": bool(has_gt),
        "auc": (float(auc) if auc is not None else None),
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "score_std": score_std,
        "seconds": seconds,
        "cube_kind": cube_k,
        "cube_path": str(cube_p),
    }

# ------------------------------- CLI / main -----------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Anomaly detection runner.")

    # ✅ 默认跑全量数据集
    p.add_argument(
        "--only",
        type=str,
        default="all",
        help=(
            "Comma list of datasets to run (e.g., Urban,Sandiego). "
            "Use 'all' to run all datasets in ds_all (default: all). "
            "You can also pass a prefix like 'CAVE_' to run all CAVE cases."
        )
    )

    p.add_argument("--methods", type=str, default="RX,RPCA_RX,SSCTV",
                   help="Comma list of methods: RX, RPCA_RX, SSCTV")
    p.add_argument("--save", type=str, default=str(RESULT_DIR),
                   help="Output directory (default: ./result)")

    # ✅ 默认顺序跑（逐个运行）
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of worker processes. 1 = sequential (default). 0 or negative = all CPU cores.")

    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(Path(args.save))

    ds_all: Dict[str, Dataset] = {
        # --- datasets with GT ---
        "Urban": Dataset("Urban", DATA_DIR / "Urban.mat", DATA_DIR / "UGt.mat", kind="mat"),
        "Sandiego": Dataset("Sandiego", DATA_DIR / "Sandiego_new.mat", DATA_DIR / "Sandiego_gt.mat", kind="mat"),

        # --- extra datasets (no GT in current .mat -> will skip ROC/AUC) ---
        "Cuprite": Dataset("Cuprite", DATA_DIR / "Cuprite.mat", None, kind="mat"),
        "DCmall": Dataset("DCmall", DATA_DIR / "DCmall.mat", None, kind="mat"),
        "KSC": Dataset("KSC", DATA_DIR / "KSC.mat", None, kind="mat"),
        "Pavia": Dataset("Pavia", DATA_DIR / "Pavia.mat", None, kind="mat"),
        "Pavia_old": Dataset("Pavia_old", DATA_DIR / "Pavia_old.mat", None, kind="mat"),
        "PaviaU": Dataset("PaviaU", DATA_DIR / "PaviaU.mat", None, kind="mat"),
        "brain_ct": Dataset("brain_ct", DATA_DIR / "brain_ct.mat", None, kind="mat"),
        "brain_mri": Dataset("brain_mri", DATA_DIR / "brain_mri.mat", None, kind="mat"),
        "chest_ct": Dataset("chest_ct", DATA_DIR / "chest_ct.mat", None, kind="mat"),
        "chest_pet": Dataset("chest_pet", DATA_DIR / "chest_pet.mat", None, kind="mat"),
    }

    # -------- add CAVE (32 cases) --------
    cave_root = DATA_DIR / "CAVE"
    cave_datasets = discover_cave_datasets(cave_root)
    if cave_datasets:
        ds_all.update(cave_datasets)
        print(f"[INFO] Discovered {len(cave_datasets)} CAVE case(s) under: {cave_root}", flush=True)
    else:
        print(f"[INFO] No CAVE cases found under: {cave_root} (skip adding CAVE).", flush=True)

    # ✅ only=all/*/空 → ds_all 全部（保持 ds_all 声明顺序）
    only = (args.only or "").strip()
    if only.lower() in ("all", "*", ""):
        wanted_keys = list(ds_all.keys())
    else:
        raw = [s.strip() for s in only.split(",") if s.strip()]
        wanted_keys: List[str] = []
        # 支持：给一个前缀（比如 CAVE_）就把所有匹配的 key 加进去
        for token in raw:
            if token.endswith("_"):
                matched = [k for k in ds_all.keys() if k.startswith(token)]
                if not matched:
                    print(f"[WARN] No dataset keys match prefix '{token}'")
                wanted_keys.extend(matched)
            else:
                wanted_keys.append(token)

        # 去重但保持顺序
        seen = set()
        wanted_keys = [k for k in wanted_keys if not (k in seen or seen.add(k))]

    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    ssctv_opts = dict(maxIter=1000, rho=1.03, tol=1e-6)

    # 选中数据集（保留顺序）
    selected: List[Dataset] = []
    for k in wanted_keys:
        if k not in ds_all:
            # 避免 choices 太长刷屏：只提示一个简短信息
            print(f"[WARN] Unknown dataset '{k}'. (Tip: use --only=all or prefix like CAVE_)")
            continue
        selected.append(ds_all[k])

    if not selected:
        print("No datasets to run.")
        return

    ds_total = len(selected)
    print("\nDatasets to run (in order):")
    for i, ds in enumerate(selected, 1):
        gt_flag = "GT=YES" if ds.gt_path else "GT=NO"
        kind_flag = f"kind={ds.kind}"
        print(f"  [{i}/{ds_total}] {ds.name} ({gt_flag}, {kind_flag})")

    # Build task list（每个方法一个 task）
    tasks: List[Tuple] = []
    task_idx = 0
    for ds_idx, ds in enumerate(selected, 1):
        out_dir = Path(args.save) / ds.name
        ensure_dir(out_dir)
        for m in methods:
            task_idx += 1
            tasks.append((
                ds.name,
                ds.kind,
                str(ds.cube_path),
                (str(ds.gt_path) if ds.gt_path else ""),
                m,
                str(out_dir),
                ssctv_opts,
                ds_idx, ds_total,
                task_idx, 0,  # task_total 先占位
            ))

    if not tasks:
        print("No tasks to run.")
        return

    task_total = len(tasks)
    tasks = [t[:-1] + (task_total,) for t in tasks]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    max_workers = (os.cpu_count() or 1) if args.jobs <= 0 else args.jobs

    print(f"\nLaunching {task_total} tasks with {max_workers} worker process(es)...")
    if max_workers == 1:
        print("Mode: SEQUENTIAL（逐个运行）\n")
    else:
        print("Mode: PARALLEL（并行，完成顺序可能与提交顺序不同）\n")

    results: List[Dict] = []
    done_by_ds = defaultdict(int)
    total_methods = len(methods)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one_method_task, *t) for t in tasks]

        # ✅ 单进程时按提交顺序逐个取结果；多进程时用 as_completed
        iterator = futs if max_workers == 1 else as_completed(futs)

        for fut in iterator:
            try:
                res = fut.result()
                results.append(res)

                ds_name = res["dataset"]
                ds_idx = res["ds_idx"]
                auc_str = _fmt(res.get("auc"), 4)
                tsec = _fmt(res.get("seconds"), 2)

                print(
                    f"  ✔ DONE  DS {ds_idx}/{ds_total} {ds_name:<20} | "
                    f"{res['method']:<8} | AUC={auc_str:<6} | "
                    f"score(mean±std)={_fmt(res['score_mean'],4)}±{_fmt(res['score_std'],4)} | "
                    f"time={tsec}s"
                )

                done_by_ds[ds_name] += 1
                if done_by_ds[ds_name] == total_methods:
                    cand = [r for r in results if r["dataset"] == ds_name and r.get("auc") is not None]
                    if cand:
                        best = max(cand, key=lambda x: x["auc"])
                        best_info = f"best={best['method']} (AUC={best['auc']:.4f})"
                    else:
                        best_info = "best=N/A (no GT)"
                    print(f"  >>> DATASET DONE  [{ds_idx}/{ds_total}] {ds_name}  ({best_info})")

            except Exception as e:
                print(f"  ✖ Task failed: {e}")

    if not results:
        print("\nAll tasks failed or no results collected.")
        return

    # ------------------ 1) 保存每个数据集 JSON ------------------
    by_ds: Dict[str, Dict[str, Optional[float]]] = {}
    for r in results:
        by_ds.setdefault(r["dataset"], {})[r["method"]] = r.get("auc")
    for ds_name, aucs in by_ds.items():
        out_dir = Path(args.save) / ds_name
        ensure_dir(out_dir)
        with open(out_dir / f"{ds_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump({"dataset": ds_name, "AUC": aucs}, f, indent=2, ensure_ascii=False)

    # ------------------ 2) 保存完整长表 CSV ------------------
    csv_all = Path(args.save) / "summary_all.csv"
    with open(csv_all, "w", encoding="utf-8") as f:
        f.write("ds_idx,ds_total,task_idx,task_total,dataset,method,has_gt,auc,score_min,score_max,score_mean,score_std,seconds,cube_kind,cube_path\n")
        for r in sorted(results, key=lambda x: (x["ds_idx"], x["method"])):
            f.write(
                f"{r['ds_idx']},{r['ds_total']},{r['task_idx']},{r['task_total']},"
                f"{r['dataset']},{r['method']},{int(r['has_gt'])},"
                f"{'' if r['auc'] is None else format(r['auc'], '.6f')},"
                f"{r['score_min']:.6f},{r['score_max']:.6f},{r['score_mean']:.6f},{r['score_std']:.6f},"
                f"{r['seconds']:.3f},{r.get('cube_kind','')},{str(r.get('cube_path','')).replace(',', ';')}\n"
            )

    # ------------------ 3) 保留旧的 AUC_summary.csv ------------------
    csv_path = Path(args.save) / "AUC_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("dataset,method,auc\n")
        for r in sorted(results, key=lambda x: (x["ds_idx"], x["method"])):
            auc = r.get("auc", None)
            auc_str = "" if auc is None else f"{auc:.6f}"
            f.write(f"{r['dataset']},{r['method']},{auc_str}\n")

    # ------------------ 4) 控制台汇总输出 ------------------
    print("\n==================== SUMMARY (LONG TABLE) ====================")
    headers = ["DS#", "Dataset", "Method", "GT", "AUC", "score_mean", "score_std", "time(s)", "kind"]
    rows = []
    for r in sorted(results, key=lambda x: (x["ds_idx"], x["method"])):
        rows.append([
            f"{r['ds_idx']}/{r['ds_total']}",
            r["dataset"],
            r["method"],
            "YES" if r["has_gt"] else "NO",
            _fmt(r.get("auc"), 4),
            _fmt(r.get("score_mean"), 4),
            _fmt(r.get("score_std"), 4),
            _fmt(r.get("seconds"), 2),
            str(r.get("cube_kind", "")),
        ])
    print(_format_table(headers, rows))

    print("\n==================== SUMMARY (AUC PIVOT) ======================")
    pivot_headers = ["Dataset"] + methods
    pivot_rows = []
    ds_order = [ds.name for ds in selected]
    auc_lookup = {(r["dataset"], r["method"]): r.get("auc") for r in results}
    for ds_name in ds_order:
        row = [ds_name]
        for m in methods:
            row.append(_fmt(auc_lookup.get((ds_name, m)), 4))
        pivot_rows.append(row)
    print(_format_table(pivot_headers, pivot_rows))

    print("\nSaved:")
    print("  -", csv_all.resolve())
    print("  -", csv_path.resolve())
    print("Done. Results are under:", Path(args.save).resolve())

if __name__ == "__main__":
    main()
