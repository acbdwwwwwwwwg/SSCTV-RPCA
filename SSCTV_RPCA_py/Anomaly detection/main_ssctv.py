from __future__ import annotations

"""
main_ssctv_repo_style.py

用途：
    参考原始 self-controler/SSCTV-RPCA 仓库的 anomaly detection 管线，
    只对 Urban 和 Sandiego 两个数据集比较：
    1) 原始 SSCTV-RPCA
    2) 新的 SSCTV-RPCA-LogDet

核心对齐点（相对 main_ssctv.py 的修正）：
    - 使用原仓库风格的数据预处理，而不是 center_align_gt + 全局 max-min。
    - Urban 采用原仓库的裁剪：M(1:80,189:288,:)
    - Sandiego 优先使用 Sandiego_new.mat；若只有 Sandiego.mat，则按原仓库删除坏波段并裁剪
    - 使用原仓库风格的逐像素谱向 L2 归一化
    - 使用原仓库 anomaly detection 的 lambda：1 / sqrt(no_rows * no_bands)
    - GT 使用原仓库风格布局：Urban 右对齐，其余默认左上放置

输出：
    默认保存到脚本所在目录（可用 --out_dir 覆盖）
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METHODS_DIR = ROOT / "methods"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

try:
    from methods.ssctv_rpca import ssctv_rpca
except Exception:
    ssctv_rpca = None

try:
    from methods.ssctv_rpca_logdet import ssctv_rpca_logdet
except Exception:
    ssctv_rpca_logdet = None


@dataclass
class DatasetPack:
    name: str
    cube: np.ndarray
    gt: np.ndarray
    notes: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cube_from_mat(path: Path) -> np.ndarray:
    md = loadmat(str(path))
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim == 3]
    if not candidates:
        raise RuntimeError(f"No 3D cube found in {path.name}. Keys: {list(md.keys())[:10]}")

    order = ["Urban", "Sandiego_new", "Sandiego", "PaviaU", "HSI", "Y", "data", "cube", "M", "X"]
    for name in order:
        for k, v in candidates:
            if name.lower() in k.lower():
                return _to_hwb_cube(v)

    _, v = max(candidates, key=lambda kv: kv[1].size)
    return _to_hwb_cube(v)


def load_gt_from_mat(path: Path) -> np.ndarray:
    md = loadmat(str(path))
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim in (1, 2)]
    if not candidates:
        raise RuntimeError(f"No GT-like array found in {path.name}. Keys: {list(md.keys())[:10]}")

    order = ["UGt", "gt", "GT", "Sandiego_gt", "PlaneGT2", "PlaneGT", "GroundTruth", "truth", "mask"]
    for name in order:
        for k, v in candidates:
            if name.lower() in k.lower():
                return np.asarray(v, dtype=np.float64)

    _, v = max(candidates, key=lambda kv: kv[1].size)
    return np.asarray(v, dtype=np.float64)


def _to_hwb_cube(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"Cube must be 3D, got shape={a.shape}")

    b_axis = int(np.argmin(a.shape))
    if b_axis == 2:
        cube = a
    elif b_axis == 0:
        cube = np.transpose(a, (1, 2, 0))
    else:
        cube = np.transpose(a, (0, 2, 1))
    return np.asarray(cube, dtype=np.float64)


def repo_pixel_l2_normalize_cube(cube: np.ndarray) -> np.ndarray:
    """模仿原 MATLAB: M = hyperConvert2D(M); M = M ./ repmat(sqrt(sum(M.^2)), size(M,1), 1)
    等价于对每个像素的谱向量做 L2 归一化。"""
    cube = np.asarray(cube, dtype=np.float64)
    denom = np.linalg.norm(cube, axis=2, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    out = cube / denom
    out[~np.isfinite(out)] = 0.0
    return out


def repo_style_lambda(cube: np.ndarray) -> float:
    h, w, b = cube.shape
    _ = h
    return float(1.0 / np.sqrt(w * b))


def build_gt_repo_style(dataset_name: str, cube: np.ndarray, gt_raw: np.ndarray) -> np.ndarray:
    h, w, _ = cube.shape
    gt = np.asarray(gt_raw)
    if gt.ndim == 1 and gt.size == h * w:
        gt = gt.reshape(h, w, order="F")
    elif gt.ndim != 2:
        raise RuntimeError(f"Unsupported GT shape for {dataset_name}: {gt.shape}")

    gt = (gt > 0).astype(np.uint8)
    gh, gw = gt.shape
    out = np.zeros((h, w), dtype=np.uint8)
    hh = min(h, gh)
    ww = min(w, gw)

    if dataset_name.upper() == "URBAN":
        # 原 MATLAB 对 Urban 使用右对齐写法：GT2(1:sa, end-sb+1:end) = GT
        out[0:hh, w - ww:w] = gt[0:hh, gw - ww:gw]
    else:
        # 原 MATLAB 对其它这里用左上放置：GT2(1:sa, 1:sb) = GT
        out[0:hh, 0:ww] = gt[0:hh, 0:ww]
    return out


def prepare_urban(data_dir: Path) -> DatasetPack:
    cube_path = data_dir / "Urban.mat"
    gt_path = data_dir / "UGt.mat"
    if not cube_path.exists():
        raise FileNotFoundError(f"Urban cube not found: {cube_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Urban GT not found: {gt_path}")

    cube = load_cube_from_mat(cube_path)
    # 原仓库：M = M(1:80,189:288,:)
    if cube.shape[0] >= 80 and cube.shape[1] >= 288:
        cube = cube[0:80, 188:288, :]
        notes = "Urban cropped to [:80, 188:288] following original MATLAB runner"
    else:
        notes = "Urban kept as-is because cube is already cropped or smaller than original crop window"

    gt_raw = load_gt_from_mat(gt_path)
    gt = build_gt_repo_style("Urban", cube, gt_raw)
    return DatasetPack("Urban", cube, gt, notes)


BAD_BANDS_ORIG = [
    1, 2, 3, 4, 5, 6,
    33, 34, 35,
    94, 95, 96, 97,
    104, 105, 106, 107, 108, 109, 110,
    153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
    221, 222, 223, 224,
]
BAD_BANDS_ORIG = sorted(set(BAD_BANDS_ORIG))


def remove_1based_bands(cube: np.ndarray, band_ids_1based: List[int]) -> np.ndarray:
    b = cube.shape[2]
    keep = [i for i in range(b) if (i + 1) not in set(band_ids_1based)]
    return cube[:, :, keep]


def prepare_sandiego(data_dir: Path) -> DatasetPack:
    new_cube = data_dir / "Sandiego_new.mat"
    new_gt = data_dir / "Sandiego_gt.mat"
    old_cube = data_dir / "Sandiego.mat"
    plane_gt2 = data_dir / "PlaneGT2.mat"

    if new_cube.exists() and new_gt.exists():
        cube = load_cube_from_mat(new_cube)
        gt_raw = load_gt_from_mat(new_gt)
        gt = build_gt_repo_style("Sandiego", cube, gt_raw)
        return DatasetPack("Sandiego", cube, gt, "Using Sandiego_new.mat + Sandiego_gt.mat directly")

    if old_cube.exists() and plane_gt2.exists():
        cube = load_cube_from_mat(old_cube)
        cube = remove_1based_bands(cube, BAD_BANDS_ORIG)
        if cube.shape[0] >= 210 and cube.shape[1] >= 380:
            # 原仓库 k==3: M = M(111:210,181:380,:)
            cube = cube[110:210, 180:380, :]
        gt_raw = load_gt_from_mat(plane_gt2)
        gt = build_gt_repo_style("Sandiego", cube, gt_raw)
        return DatasetPack("Sandiego", cube, gt, "Using Sandiego.mat with original band removal and crop [110:210, 180:380]")

    raise FileNotFoundError(
        "Sandiego data not found. Expected either (Sandiego_new.mat + Sandiego_gt.mat) or (Sandiego.mat + PlaneGT2.mat)."
    )


def dataset_packs(data_dir: Path) -> List[DatasetPack]:
    return [prepare_urban(data_dir), prepare_sandiego(data_dir)]


def roc_curve_pf_pd(scores: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = scores.reshape(-1).astype(float)
    y = (gt.reshape(-1) > 0).astype(np.uint8)
    order = np.argsort(-s)
    y = y[order]
    p = int(np.sum(y == 1))
    n = int(np.sum(y == 0))
    if p == 0 or n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)
    pd = tp / p
    pf = fp / n
    pf = np.concatenate(([0.0], pf, [1.0]))
    pd = np.concatenate(([0.0], pd, [1.0]))
    return pf, pd


def auc_trapz(pf: np.ndarray, pd: np.ndarray) -> float:
    order = np.argsort(pf)
    return float(np.trapz(pd[order], pf[order]))


def _score_from_residual(residual: np.ndarray, cube: np.ndarray) -> np.ndarray:
    h, w, b = cube.shape
    r = residual
    if r is None:
        raise RuntimeError("Residual/Sparse component is None.")

    if r.ndim == 3:
        if r.shape == (h, w, b):
            return np.linalg.norm(r, axis=2)
        for axes in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            rt = np.transpose(r, axes)
            if rt.shape == (h, w, b):
                return np.linalg.norm(rt, axis=2)

    if r.ndim == 2:
        if r.shape == (h, w):
            return np.asarray(r, dtype=np.float64)
        if r.shape == (w, h):
            return np.asarray(r.T, dtype=np.float64)
        if r.shape[1] == b:
            return np.linalg.norm(r, axis=1).reshape(h, w, order="F")
        if r.shape[0] == b:
            return np.linalg.norm(r, axis=0).reshape(h, w, order="F")
        if r.size == h * w * b:
            rt = r.reshape(h, w, b, order="F")
            return np.linalg.norm(rt, axis=2)
        if r.size == h * w:
            return r.reshape(h, w, order="F")

    if r.ndim == 1 and r.size == h * w:
        return r.reshape(h, w, order="F")

    raise RuntimeError(f"Cannot infer score map from residual shape {r.shape} for cube {(h, w, b)}.")


def save_heatmap(score: np.ndarray, out_png: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(score, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_binary_map(img: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(4, 4), dpi=200)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_roc(pf: np.ndarray, pd: np.ndarray, auc: float, title: str, out_png: Path) -> None:
    plt.figure(figsize=(4, 4), dpi=160)
    plt.plot(pf, pd, lw=2)
    plt.xlabel("PF (False Positive Rate)")
    plt.ylabel("PD (True Positive Rate)")
    plt.title(f"ROC • {title} • AUC={auc:.4f}")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def detector_SSCTV(cube: np.ndarray, lam: float, opts: Optional[Dict] = None) -> np.ndarray:
    if ssctv_rpca is None:
        raise RuntimeError("methods/ssctv_rpca.py not found or import failed.")
    local_opts = {'lambda_': lam, 'lambdaVal': lam, 'lambda_val': lam, 'lambda': lam, 'maxIter': 1000, 'rho': 1.03, 'tol': 1e-6}
    if opts:
        local_opts.update(opts)
    try:
        _, e = ssctv_rpca(cube, opts=local_opts)
    except TypeError:
        _, e = ssctv_rpca(cube, **local_opts)
    return _score_from_residual(e, cube)


def detector_SSCTV_LOGDET(cube: np.ndarray, lam: float, opts: Optional[Dict] = None) -> np.ndarray:
    if ssctv_rpca_logdet is None:
        raise RuntimeError("methods/ssctv_rpca_logdet.py not found or import failed.")
    local_opts = dict(
        lambda1=lam,
        lambda_1=lam,
        lambda2=0.1 * lam,
        lambda_2=0.1 * lam,
        alpha1=1.0,
        alpha2=1.0,
        maxIter=1000,
        rho=1.03,
        tol=1e-6,
    )
    if opts:
        local_opts.update(opts)
    try:
        _, s = ssctv_rpca_logdet(cube, opts=local_opts)
    except TypeError:
        _, s = ssctv_rpca_logdet(cube, **local_opts)
    return _score_from_residual(s, cube)


def run_one_method(dataset: DatasetPack, method: str, out_dir: Path, method_opts: Optional[Dict] = None) -> Dict:
    t0 = time.perf_counter()
    cube = repo_pixel_l2_normalize_cube(dataset.cube)
    lam = repo_style_lambda(cube)

    if method.upper() == "SSCTV":
        score = detector_SSCTV(cube, lam, method_opts)
        method_name = "SSCTV"
    elif method.upper() in ("SSCTV_LOGDET", "SSCTV-LOGDET", "LOGDET"):
        score = detector_SSCTV_LOGDET(cube, lam, method_opts)
        method_name = "SSCTV_LOGDET"
    else:
        raise RuntimeError(f"Unknown method: {method}")

    score = np.asarray(score, dtype=np.float64)
    smin = float(np.nanmin(score))
    smax = float(np.nanmax(score))
    score_vis = (score - smin) / (smax - smin) if smax > smin else np.zeros_like(score)

    base = f"{dataset.name}_{method_name}"
    np.save(out_dir / f"{base}_score.npy", score.astype(np.float32))
    save_heatmap(score_vis, out_dir / f"{base}_score.png")

    pf, pd = roc_curve_pf_pd(score, dataset.gt)
    auc = auc_trapz(pf, pd)
    save_roc(pf, pd, auc, f"{dataset.name} • {method_name}", out_dir / f"{base}_ROC.png")

    sec = float(time.perf_counter() - t0)
    return {
        "dataset": dataset.name,
        "method": method_name,
        "auc": float(auc),
        "time_sec": sec,
        "score_min": float(np.nanmin(score)),
        "score_max": float(np.nanmax(score)),
        "score_mean": float(np.nanmean(score)),
        "score_std": float(np.nanstd(score)),
        "lambda_used": lam,
        "notes": dataset.notes,
    }


def save_summary(results: List[Dict], out_dir: Path) -> None:
    json_path = out_dir / "main_ssctv_auc_summary.json"
    csv_path = out_dir / "main_ssctv_auc_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "method", "auc", "time_sec", "score_min", "score_max",
            "score_mean", "score_std", "lambda_used", "notes"
        ])
        for r in results:
            w.writerow([
                r["dataset"], r["method"], f"{r['auc']:.6f}", f"{r['time_sec']:.6f}",
                f"{r['score_min']:.6f}", f"{r['score_max']:.6f}", f"{r['score_mean']:.6f}",
                f"{r['score_std']:.6f}", f"{r['lambda_used']:.8f}", r["notes"]
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare SSCTV and SSCTV_LOGDET on Urban and Sandiego using original repository-style anomaly detection preprocessing."
    )
    p.add_argument("--out_dir", type=str, default=str(ROOT), help="Output directory. Default: script folder")
    p.add_argument("--ssctv_maxIter", type=int, default=1000)
    p.add_argument("--ssctv_rho", type=float, default=1.03)
    p.add_argument("--ssctv_tol", type=float, default=1e-6)
    p.add_argument("--logdet_maxIter", type=int, default=1000)
    p.add_argument("--logdet_rho", type=float, default=1.03)
    p.add_argument("--logdet_tol", type=float, default=1e-6)
    p.add_argument("--alpha1", type=float, default=1.0)
    p.add_argument("--alpha2", type=float, default=1.0)
    p.add_argument("--lambda2_ratio", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    packs = dataset_packs(DATA_DIR)
    for ds in packs:
        save_binary_map(ds.gt, out_dir / f"{ds.name}_GT_repo_style.png")

    results: List[Dict] = []
    for ds in packs:
        print(f"\n===== Dataset: {ds.name} =====", flush=True)
        print(f"[INFO] {ds.notes}", flush=True)

        lam = repo_style_lambda(ds.cube)
        ssctv_opts = {
            "lambda": lam,
            "lambda_": lam,
            "lambdaVal": lam,
            "lambda_val": lam,
            "maxIter": args.ssctv_maxIter,
            "rho": args.ssctv_rho,
            "tol": args.ssctv_tol,
        }
        logdet_opts = {
            "lambda1": lam,
            "lambda_1": lam,
            "lambda2": args.lambda2_ratio * lam,
            "lambda_2": args.lambda2_ratio * lam,
            "alpha1": args.alpha1,
            "alpha2": args.alpha2,
            "maxIter": args.logdet_maxIter,
            "rho": args.logdet_rho,
            "tol": args.logdet_tol,
        }

        for method, opts in (("SSCTV", ssctv_opts), ("SSCTV_LOGDET", logdet_opts)):
            print(f"[RUN ] {ds.name} | {method} | lambda={lam:.8f}", flush=True)
            res = run_one_method(ds, method, out_dir, opts)
            results.append(res)
            print(f"[DONE] {ds.name} | {method:12s} | AUC={res['auc']:.6f} | time={res['time_sec']:.2f}s", flush=True)

    save_summary(results, out_dir)

    print("\n========== AUC SUMMARY ==========")
    for ds_name in ("Urban", "Sandiego"):
        for r in [x for x in results if x["dataset"] == ds_name]:
            print(f"{r['dataset']:10s} | {r['method']:12s} | AUC={r['auc']:.6f} | lambda={r['lambda_used']:.8f} | time={r['time_sec']:.2f}s")

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()