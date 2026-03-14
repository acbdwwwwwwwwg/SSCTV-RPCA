from __future__ import annotations

"""
main_ssctv.py

用于在 Anomaly detection 目录下比较：
1) 原始 SSCTV-RPCA
2) 新的 ssctv_rpca_logdet.py

特点：
- 默认跑原仓库 main.py 中注册的全部数据集
- 支持 --dataset 指定一个或多个数据集
- 支持 --model 指定跑 SSCTV / SSCTV_LOGDET / BOTH
- GT 文件不存在时自动降级为无 GT 模式（仍输出 score / heatmap）
- 支持直接传 --cube / --gt 路径，绕过注册表
- 支持自动发现 data/CAVE 下的 case（与原仓库 main.py 思路一致）

建议放置：
    SSCTV_RPCA_py/Anomaly detection/main_ssctv.py
"""

import argparse
import csv
import json
import os
import re
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

# --------------------------- threading hygiene --------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------ paths -----------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METHODS_DIR = ROOT / "methods"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(METHODS_DIR) not in sys.path:
    sys.path.insert(0, str(METHODS_DIR))

# ------------------------------ methods ---------------------------------------
try:
    from methods.ssctv_rpca import ssctv_rpca
except Exception:
    ssctv_rpca = None

try:
    from methods.ssctv_rpca_logdet import ssctv_rpca_logdet
except Exception:
    ssctv_rpca_logdet = None


@dataclass
class Dataset:
    name: str
    cube_path: Path
    gt_path: Optional[Path] = None
    kind: str = "mat"   # "mat" or "cave"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------ I/O helpers -----------------------------------
def load_cube_from_mat(path: Path) -> np.ndarray:
    """从 .mat 加载 3D 立方体。按原仓库的常见 key 顺序猜测变量名。"""
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
    """从 .mat 加载 GT（1/0）。尽量容忍不同命名与形状。"""
    md = loadmat(str(path))
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim in (1, 2)]
    if not candidates:
        raise RuntimeError(f"No GT-like array found in {path.name}. Keys: {list(md.keys())[:10]}")

    order = ["UGt", "gt", "GT", "Sandiego_gt", "GroundTruth", "truth", "mask"]
    for name in order:
        for k, v in candidates:
            if name.lower() in k.lower():
                return np.asarray(v, dtype=np.float64)

    _, v = max(candidates, key=lambda kv: kv[1].size)
    return np.asarray(v, dtype=np.float64)


def _to_hwb_cube(arr: np.ndarray) -> np.ndarray:
    A = np.asarray(arr)
    if A.ndim != 3:
        raise ValueError(f"Cube must be 3D, got shape={A.shape}")

    # 经验：波段维通常最小；若最后一维最小，则直接返回
    shape = A.shape
    b_axis = int(np.argmin(shape))
    if b_axis == 2:
        cube = A
    elif b_axis == 0:
        cube = np.transpose(A, (1, 2, 0))
    else:
        cube = np.transpose(A, (0, 2, 1))
    return np.asarray(cube, dtype=np.float64)


def _imread_any(path: Path) -> np.ndarray:
    try:
        import imageio.v2 as imageio  # type: ignore
        img = imageio.imread(str(path))
    except Exception:
        from PIL import Image  # type: ignore
        img = np.array(Image.open(path))

    if img.ndim == 3:
        img = img[..., 0]
    return np.asarray(img, dtype=np.float64)


def _find_cave_inner_dir(case_root: Path) -> Path:
    """适配 data/CAVE/case/case/*.png 或 data/CAVE/case/*.png 两种结构。"""
    nested = case_root / case_root.name
    if nested.exists() and nested.is_dir():
        return nested
    return case_root


def load_cube_from_cave_case(case_dir: Path, expected_bands: int = 31) -> np.ndarray:
    case_dir = _find_cave_inner_dir(case_dir)
    if not case_dir.exists():
        raise RuntimeError(f"CAVE case dir not found: {case_dir}")

    pngs = list(case_dir.glob("*_ms_*.png"))
    if not pngs:
        pngs = list(case_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG bands found under: {case_dir}")

    def _band_index(p: Path) -> int:
        m = re.search(r"_([0-9]{1,3})\.png$", p.name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        return 10**9

    pngs_sorted = sorted(pngs, key=lambda p: (_band_index(p), p.name.lower()))
    has_index = [p for p in pngs_sorted if _band_index(p) < 10**9]
    if len(has_index) >= 2:
        pngs_sorted = has_index

    if expected_bands is not None and len(pngs_sorted) != expected_bands:
        print(f"[WARN] CAVE case {case_dir} has {len(pngs_sorted)} PNGs (expected {expected_bands}); load all found.", flush=True)

    bands = [_imread_any(p) for p in pngs_sorted]
    h, w = bands[0].shape
    for i, b in enumerate(bands, 1):
        if b.shape != (h, w):
            raise RuntimeError(f"Inconsistent band shape in {case_dir}: band#{i} {b.shape} vs {(h, w)}")
    cube = np.stack(bands, axis=2)
    return np.asarray(cube, dtype=np.float64)


# ------------------------------ dataset registry ------------------------------
def discover_cave_datasets(cave_root: Path) -> Dict[str, Dataset]:
    ds: Dict[str, Dataset] = {}
    if not cave_root.exists() or not cave_root.is_dir():
        return ds

    case_dirs = [p for p in cave_root.iterdir() if p.is_dir()]
    for p in sorted(case_dirs, key=lambda x: x.name.lower()):
        key = f"CAVE_{p.name}"
        ds[key] = Dataset(name=key, cube_path=p, gt_path=None, kind="cave")
    return ds


def build_dataset_registry() -> Dict[str, Dataset]:
    """同步原仓库 main.py 的 ds_all 注册逻辑。"""
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

    cave_root = DATA_DIR / "CAVE"
    cave_datasets = discover_cave_datasets(cave_root)
    if cave_datasets:
        ds_all.update(cave_datasets)
        print(f"[INFO] Discovered {len(cave_datasets)} CAVE case(s) under: {cave_root}", flush=True)
    else:
        print(f"[INFO] No CAVE cases found under: {cave_root} (skip adding CAVE).", flush=True)
    return ds_all


# ------------------------------ metric helpers --------------------------------
def minmax_normalize_cube(cube: np.ndarray) -> np.ndarray:
    """按波段 min-max 到 [0,1]（与原仓库 main.py 保持一致）。"""
    cube = np.asarray(cube, dtype=np.float64)
    H, W, B = cube.shape
    out = np.empty_like(cube)
    for b in range(B):
        band = cube[:, :, b]
        mn = float(np.nanmin(band))
        mx = float(np.nanmax(band))
        if mx > mn:
            out[:, :, b] = (band - mn) / (mx - mn)
        else:
            out[:, :, b] = 0.0
    return out


def center_align_gt(cube: np.ndarray, gt: np.ndarray) -> np.ndarray:
    H, W, _ = cube.shape
    gt = np.asarray(gt)

    if gt.ndim == 1 and gt.size == H * W:
        return (gt.reshape(H, W, order="F") > 0).astype(np.uint8)

    if gt.ndim == 2:
        gh, gw = gt.shape
        if (gh, gw) == (H, W):
            return (gt > 0).astype(np.uint8)
        out = np.zeros((H, W), dtype=np.uint8)
        h = min(H, gh)
        w = min(W, gw)
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        gy0 = (gh - h) // 2
        gx0 = (gw - w) // 2
        out[y0:y0 + h, x0:x0 + w] = (gt[gy0:gy0 + h, gx0:gx0 + w] > 0).astype(np.uint8)
        return out

    raise RuntimeError(f"Unsupported GT shape: {gt.shape}")


def save_heatmap(score: np.ndarray, out_png: Path, cmap: str = "viridis") -> None:
    plt.figure(figsize=(5, 4), dpi=160)
    plt.imshow(score, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def roc_curve_pf_pd(scores: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = scores.reshape(-1).astype(float)
    y = (gt.reshape(-1) > 0).astype(np.uint8)

    order = np.argsort(-s)
    y = y[order]

    P = max(int(np.sum(y == 1)), 1)
    N = max(int(np.sum(y == 0)), 1)

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    pd = tp / P
    pf = fp / N

    pd = np.concatenate([[0.0], pd])
    pf = np.concatenate([[0.0], pf])
    return pf, pd


def auc_trapz(pf: np.ndarray, pd: np.ndarray) -> float:
    order = np.argsort(pf)
    return float(np.trapz(pd[order], pf[order]))


def _score_from_residual(residual: np.ndarray, cube: np.ndarray) -> np.ndarray:
    H, W, B = cube.shape
    R = residual
    if R is None:
        raise RuntimeError("Residual/Sparse component is None.")

    if R.ndim == 3:
        if R.shape == (H, W, B):
            return np.linalg.norm(R, axis=2)
        for axes in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            Rt = np.transpose(R, axes)
            if Rt.shape == (H, W, B):
                return np.linalg.norm(Rt, axis=2)

    if R.ndim == 2:
        if R.shape == (H, W):
            return R
        if R.shape == (W, H):
            return R.T
        if R.shape[1] == B:
            return np.linalg.norm(R, axis=1).reshape(H, W, order="F")
        if R.shape[0] == B:
            return np.linalg.norm(R, axis=0).reshape(H, W, order="F")
        if R.size == H * W * B:
            Rt = R.reshape(H, W, B, order="F")
            return np.linalg.norm(Rt, axis=2)
        if R.size == H * W:
            return R.reshape(H, W, order="F")

    if R.ndim == 1 and R.size == H * W:
        return R.reshape(H, W, order="F")

    raise RuntimeError(f"Cannot infer score map from residual shape {R.shape} for cube {(H, W, B)}.")


# ------------------------------ detectors -------------------------------------
def detector_SSCTV(cube: np.ndarray, lam: float, opts: Optional[Dict] = None) -> np.ndarray:
    if ssctv_rpca is None:
        raise RuntimeError("methods/ssctv_rpca.py not found or import failed.")
    local_opts = dict(lambda_=lam, maxIter=1000, rho=1.03, tol=1e-6)
    if opts:
        local_opts.update(opts)
    try:
        _, E = ssctv_rpca(cube, opts=local_opts)
    except TypeError:
        _, E = ssctv_rpca(cube, **local_opts)
    return _score_from_residual(E, cube)


def detector_SSCTV_LOGDET(cube: np.ndarray, lam: float, opts: Optional[Dict] = None) -> np.ndarray:
    if ssctv_rpca_logdet is None:
        raise RuntimeError("methods/ssctv_rpca_logdet.py not found or import failed.")
    local_opts = dict(
        lambda1=lam,
        lambda2=0.1 * lam,
        alpha1=1.0,
        alpha2=1.0,
        maxIter=1000,
        rho=1.03,
        tol=1e-6,
    )
    if opts:
        local_opts.update(opts)
    try:
        _, S = ssctv_rpca_logdet(cube, opts=local_opts)
    except TypeError:
        _, S = ssctv_rpca_logdet(cube, **local_opts)
    return _score_from_residual(S, cube)


# ------------------------------ running ---------------------------------------
def load_dataset_cube_and_gt(ds: Dataset) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if ds.kind == "mat":
        if not ds.cube_path.exists():
            raise FileNotFoundError(f"Cube file not found: {ds.cube_path}")
        cube = load_cube_from_mat(ds.cube_path)
        gt = None
        if ds.gt_path is not None:
            if ds.gt_path.exists():
                gt = load_gt_from_mat(ds.gt_path)
            else:
                print(f"[WARN] GT file not found for {ds.name}: {ds.gt_path}; run without GT.", flush=True)
        return cube, gt

    if ds.kind == "cave":
        cube = load_cube_from_cave_case(ds.cube_path)
        return cube, None

    raise RuntimeError(f"Unsupported dataset kind: {ds.kind}")


def run_one_method(dataset_name: str,
                   cube: np.ndarray,
                   gt: Optional[np.ndarray],
                   method: str,
                   out_dir: Path,
                   method_opts: Optional[Dict] = None) -> Dict:
    t0 = time.perf_counter()

    cube = minmax_normalize_cube(cube)
    H, W, _ = cube.shape
    lam = 1.0 / np.sqrt(H * W)

    gt_aligned = center_align_gt(cube, gt) if gt is not None else None
    has_gt = gt_aligned is not None

    method_u = method.upper()
    if method_u == "SSCTV":
        score = detector_SSCTV(cube, lam, method_opts)
    elif method_u in ("SSCTV_LOGDET", "SSCTV-LOGDET", "LOGDET"):
        score = detector_SSCTV_LOGDET(cube, lam, method_opts)
        method_u = "SSCTV_LOGDET"
    else:
        raise RuntimeError(f"Unknown method: {method}")

    score = score.astype(np.float64, copy=False)
    smin = float(np.nanmin(score))
    smax = float(np.nanmax(score))
    smean = float(np.nanmean(score))
    sstd = float(np.nanstd(score))
    score_vis = (score - smin) / (smax - smin) if smax > smin else np.zeros_like(score)

    ensure_dir(out_dir)
    base = f"{dataset_name}_{method_u}"
    np.save(out_dir / f"{base}_score.npy", score.astype(np.float32))
    save_heatmap(score_vis, out_dir / f"{base}_score.png")

    auc = None
    if has_gt:
        pf, pd = roc_curve_pf_pd(score, gt_aligned)
        auc = auc_trapz(pf, pd)
        plt.figure(figsize=(4.5, 4.0), dpi=160)
        plt.plot(pf, pd, lw=2)
        plt.xlabel("PF (False Positive Rate)")
        plt.ylabel("PD (True Positive Rate)")
        plt.title(f"ROC • {dataset_name} • {method_u} • AUC={auc:.4f}")
        plt.grid(True, ls="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_dir / f"{base}_ROC.png")
        plt.close()

    sec = float(time.perf_counter() - t0)
    return {
        "dataset": dataset_name,
        "method": method_u,
        "auc": (float(auc) if auc is not None else None),
        "time_sec": sec,
        "score_min": smin,
        "score_max": smax,
        "score_mean": smean,
        "score_std": sstd,
        "out_dir": str(out_dir),
    }


def resolve_direct_dataset(args) -> Optional[List[Dataset]]:
    if not args.cube:
        return None
    cube_path = Path(args.cube).expanduser().resolve()
    gt_path = Path(args.gt).expanduser().resolve() if args.gt else None
    name = args.dataset.strip() if args.dataset.strip() else cube_path.stem
    return [Dataset(name=name, cube_path=cube_path, gt_path=gt_path, kind="mat")]


def resolve_registry_datasets(args, ds_all: Dict[str, Dataset]) -> List[Dataset]:
    token_str = (args.dataset or "all").strip()
    if token_str.lower() in ("all", "*", ""):
        wanted_keys = list(ds_all.keys())
    else:
        raw = [s.strip() for s in token_str.split(",") if s.strip()]
        wanted_keys: List[str] = []
        for token in raw:
            if token.endswith("_"):
                matched = [k for k in ds_all.keys() if k.startswith(token)]
                if not matched:
                    print(f"[WARN] No dataset keys match prefix '{token}'")
                wanted_keys.extend(matched)
            else:
                wanted_keys.append(token)

        seen = set()
        wanted_keys = [k for k in wanted_keys if not (k in seen or seen.add(k))]

    selected: List[Dataset] = []
    for k in wanted_keys:
        if k not in ds_all:
            print(f"[WARN] Unknown dataset '{k}' (skip).")
            continue
        selected.append(ds_all[k])

    if not selected:
        avail = ", ".join(ds_all.keys())
        raise SystemExit(f"No valid datasets selected. Available: {avail}")
    return selected


# ------------------------------ CLI -------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare original SSCTV-RPCA and SSCTV-RPCA-LogDet for anomaly detection.")
    p.add_argument("--dataset", type=str, default="all",
                   help="Dataset name(s) in registry. Use 'all' for all datasets, comma list for multiple, or prefix ending '_' such as CAVE_.")
    p.add_argument("--cube", type=str, default="", help="Direct cube .mat path; if set, registry is bypassed.")
    p.add_argument("--gt", type=str, default="", help="Optional GT .mat path for direct mode.")
    p.add_argument("--model", type=str, default="BOTH", choices=["SSCTV", "SSCTV_LOGDET", "BOTH"],
                   help="Which model to run.")
    p.add_argument("--out_dir", type=str, default="./result", help="Output directory.")

    # 原版 SSCTV 参数
    p.add_argument("--ssctv_maxIter", type=int, default=1000)
    p.add_argument("--ssctv_rho", type=float, default=1.03)
    p.add_argument("--ssctv_tol", type=float, default=1e-6)
    p.add_argument("--ssctv_lambda", type=float, default=None,
                   help="Override default lambda=1/sqrt(HW) for original SSCTV.")

    # LogDet 版参数
    p.add_argument("--logdet_maxIter", type=int, default=1000)
    p.add_argument("--logdet_rho", type=float, default=1.03)
    p.add_argument("--logdet_tol", type=float, default=1e-6)
    p.add_argument("--lambda1", type=float, default=None,
                   help="Override default lambda1=1/sqrt(HW) for SSCTV_LOGDET.")
    p.add_argument("--lambda2", type=float, default=None,
                   help="Override default lambda2=0.1*lambda1 for SSCTV_LOGDET.")
    p.add_argument("--alpha1", type=float, default=1.0)
    p.add_argument("--alpha2", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()

    ds_all = build_dataset_registry()
    selected = resolve_direct_dataset(args)
    if selected is None:
        selected = resolve_registry_datasets(args, ds_all)

    methods = ["SSCTV", "SSCTV_LOGDET"] if args.model == "BOTH" else [args.model]
    out_root = ensure_dir(Path(args.out_dir).expanduser().resolve())

    print(f"[INFO] Selected datasets: {len(selected)}")
    print(f"[INFO] Methods: {', '.join(methods)}")
    print(f"[INFO] Output root: {out_root}")

    all_results: List[Dict] = []

    for idx, ds in enumerate(selected, start=1):
        print(f"\n===== [{idx}/{len(selected)}] Dataset: {ds.name} =====", flush=True)
        try:
            cube, gt = load_dataset_cube_and_gt(ds)
        except Exception as e:
            print(f"[ERROR] Failed to load dataset {ds.name}: {e}", flush=True)
            continue

        ds_out_dir = ensure_dir(out_root / ds.name)

        for method in methods:
            if method == "SSCTV":
                method_opts = {
                    "maxIter": args.ssctv_maxIter,
                    "rho": args.ssctv_rho,
                    "tol": args.ssctv_tol,
                }
                if args.ssctv_lambda is not None:
                    method_opts["lambda_"] = args.ssctv_lambda
            else:
                method_opts = {
                    "maxIter": args.logdet_maxIter,
                    "rho": args.logdet_rho,
                    "tol": args.logdet_tol,
                    "alpha1": args.alpha1,
                    "alpha2": args.alpha2,
                }
                if args.lambda1 is not None:
                    method_opts["lambda1"] = args.lambda1
                if args.lambda2 is not None:
                    method_opts["lambda2"] = args.lambda2

            print(f"[RUN ] {ds.name} | {method}", flush=True)
            try:
                res = run_one_method(
                    dataset_name=ds.name,
                    cube=cube,
                    gt=gt,
                    method=method,
                    out_dir=ds_out_dir,
                    method_opts=method_opts,
                )
                all_results.append(res)
                auc_str = "N/A" if res["auc"] is None else f"{res['auc']:.6f}"
                print(f"[DONE] {ds.name} | {method:12s} | AUC={auc_str} | time={res['time_sec']:.2f}s", flush=True)
            except Exception as e:
                print(f"[ERROR] {ds.name} | {method}: {e}", flush=True)

    # 汇总保存
    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(out_root / "summary_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "auc", "time_sec", "score_min", "score_max", "score_mean", "score_std", "out_dir"])
        for r in all_results:
            w.writerow([
                r["dataset"], r["method"], r["auc"], f"{r['time_sec']:.6f}",
                f"{r['score_min']:.6f}", f"{r['score_max']:.6f}",
                f"{r['score_mean']:.6f}", f"{r['score_std']:.6f}", r["out_dir"]
            ])

    # AUC 透视表
    pivot: Dict[str, Dict[str, Optional[float]]] = {}
    for r in all_results:
        pivot.setdefault(r["dataset"], {})[r["method"]] = r["auc"]

    with open(out_root / "summary_auc_pivot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "SSCTV", "SSCTV_LOGDET"])
        for ds_name in sorted(pivot.keys()):
            row = pivot[ds_name]
            w.writerow([ds_name, row.get("SSCTV"), row.get("SSCTV_LOGDET")])

    print("\n========== SUMMARY ==========")
    if not all_results:
        print("No successful runs.")
    else:
        for r in all_results:
            auc_str = "N/A" if r["auc"] is None else f"{r['auc']:.6f}"
            print(f"{r['dataset']:20s} | {r['method']:12s} | AUC={auc_str:>10s} | time={r['time_sec']:.2f}s")
    print(f"Saved to: {out_root}")


if __name__ == "__main__":
    main()
