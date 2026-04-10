from __future__ import annotations

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


def candidate_data_dirs() -> List[Path]:
    dirs = [
        ROOT / "data",
        ROOT,
        Path.cwd() / "data",
        Path.cwd(),
    ]
    out: List[Path] = []
    seen = set()
    for d in dirs:
        try:
            key = str(d.resolve())
        except Exception:
            key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def find_data_file(name: str) -> Path:
    for d in candidate_data_dirs():
        p = d / name
        if p.exists():
            return p
    return candidate_data_dirs()[0] / name


def load_mat(path: Path) -> Dict:
    return loadmat(str(path))


def load_cube_from_mat(path: Path) -> np.ndarray:
    md = load_mat(path)
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim == 3]
    if not candidates:
        raise RuntimeError(f"No 3D cube found in {path.name}. Keys: {list(md.keys())[:10]}")

    order = ["Urban", "Sandiego_new", "Sandiego", "data", "PaviaU", "HSI", "Y", "cube", "M", "X"]
    for name in order:
        for k, v in candidates:
            if name.lower() in k.lower():
                return _to_hwb_cube(v)

    _, v = max(candidates, key=lambda kv: kv[1].size)
    return _to_hwb_cube(v)


def load_gt_from_mat(path: Path) -> np.ndarray:
    md = load_mat(path)
    candidates = [(k, v) for k, v in md.items() if isinstance(v, np.ndarray) and v.ndim in (1, 2)]
    if not candidates:
        raise RuntimeError(f"No GT-like array found in {path.name}. Keys: {list(md.keys())[:10]}")

    order = ["UGt", "PlaneGT2", "PlaneGT", "Sandiego_gt", "map", "gt", "GT", "GroundTruth", "truth", "mask"]
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

    s0, s1, s2 = a.shape

    if s2 >= s0 and s2 >= s1:
        cube = a
    elif s0 >= s1 and s0 >= s2:
        cube = np.transpose(a, (1, 2, 0))
    elif s1 >= s0 and s1 >= s2:
        cube = np.transpose(a, (0, 2, 1))
    else:
        cube = a

    return np.asarray(cube, dtype=np.float64)


def repo_pixel_l2_normalize_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float64)
    denom = np.linalg.norm(cube, axis=2, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    out = cube / denom
    out[~np.isfinite(out)] = 0.0
    return out


def repo_style_lambda(cube: np.ndarray) -> float:
    _, w, b = cube.shape
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
        out[0:hh, w - ww:w] = gt[0:hh, gw - ww:gw]
    else:
        out[0:hh, 0:ww] = gt[0:hh, 0:ww]
    return out


def prepare_urban(data_dir: Path) -> DatasetPack:
    _ = data_dir
    cube_path = find_data_file("Urban.mat")
    gt_path = find_data_file("UGt.mat")
    if not cube_path.exists():
        raise FileNotFoundError(f"Urban cube not found: {cube_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Urban GT not found: {gt_path}")

    cube = load_cube_from_mat(cube_path)
    if cube.shape[0] >= 80 and cube.shape[1] >= 288:
        cube = cube[0:80, 188:288, :]
        notes = "Urban cropped to [:80, 188:288] following original MATLAB runner"
    else:
        notes = "Urban kept as-is because cube is already cropped or smaller than original crop window"

    gt_raw = load_gt_from_mat(gt_path)
    gt = build_gt_repo_style("Urban", cube, gt_raw)
    return DatasetPack("Urban", cube, gt, notes)


# self-controler/SSCTV-RPCA 原仓库 Anomaly detection/main.m 中的删带方式
BAD_BANDS_REPO_K1 = sorted(set([
    1, 2, 3, 4, 5, 6,
    33, 34, 35,
    94, 95, 96, 97,
    104, 105, 106, 107, 108, 109, 110,
    153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
    221, 222, 223, 224,
]))

# 根据上传的 Sandiego.mat(100x100x224) 与 Sandiego_new.mat(100x100x189) 的对应关系反推的删带方式
BAD_BANDS_UPLOADED_TO_NEW189 = sorted(set([
    1, 2, 3, 4, 5, 6,
    33, 34, 35,
    97,
    107, 108, 109, 110, 111, 112, 113,
    153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
    221, 222, 223, 224,
]))


def remove_1based_bands(cube: np.ndarray, band_ids_1based: List[int]) -> np.ndarray:
    band_ids = set(int(x) for x in band_ids_1based)
    keep = [i for i in range(cube.shape[2]) if (i + 1) not in band_ids]
    return cube[:, :, keep]


def _maybe_get_embedded_map(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    md = load_mat(path)
    v = md.get("map", None)
    if isinstance(v, np.ndarray) and v.ndim == 2:
        return np.asarray(v, dtype=np.float64)
    return None


def _note_gt_preference(has_plane: bool, has_newgt: bool) -> str:
    if has_plane:
        return "GT source: PlaneGT/map preferred for repo alignment"
    if has_newgt:
        return "GT source: Sandiego_gt fallback"
    return "GT source unavailable"


def prepare_sandiego(data_dir: Path, mode: str = "best_effort_repo") -> DatasetPack:
    _ = data_dir
    sandiego_mat = find_data_file("Sandiego.mat")
    sandiego_new = find_data_file("Sandiego_new.mat")
    sandiego_gt = find_data_file("Sandiego_gt.mat")
    plane_gt = find_data_file("PlaneGT.mat")

    old_cube = load_cube_from_mat(sandiego_mat) if sandiego_mat.exists() else None
    new_cube = load_cube_from_mat(sandiego_new) if sandiego_new.exists() else None

    gt_plane = load_gt_from_mat(plane_gt) if plane_gt.exists() else None
    embedded_map = _maybe_get_embedded_map(sandiego_mat)
    gt_new = load_gt_from_mat(sandiego_gt) if sandiego_gt.exists() else None

    gt_repo = gt_plane if gt_plane is not None else embedded_map
    gt_fallback = gt_new if gt_new is not None else gt_repo
    _ = gt_fallback

    if mode == "repo_mainm_186":
        if old_cube is None or gt_repo is None:
            raise FileNotFoundError(
                "repo_mainm_186 模式需要 Sandiego.mat 和 PlaneGT.mat(或 Sandiego.mat 内置 map)。"
            )
        cube = remove_1based_bands(old_cube, BAD_BANDS_REPO_K1)
        if cube.shape[0] >= 150 and cube.shape[1] >= 150:
            cube = cube[0:150, 0:150, :]
            note = "Sandiego via original repo k==1 style: remove repo bands + crop [:150, :150] + PlaneGT"
        else:
            note = (
                "Sandiego via repo_mainm_186 best effort: current Sandiego.mat is already smaller than 150x150, "
                "so only repo bad-band removal is applied"
            )
        gt = build_gt_repo_style("Sandiego", cube, gt_repo)
        return DatasetPack("Sandiego", cube, gt, note)

    if mode == "new189_planegt":
        if new_cube is None:
            if old_cube is None:
                raise FileNotFoundError("new189_planegt 模式需要 Sandiego_new.mat，或可从 Sandiego.mat 构造。")
            new_cube = remove_1based_bands(old_cube, BAD_BANDS_UPLOADED_TO_NEW189)
        gt_src = gt_repo if gt_repo is not None else gt_new
        if gt_src is None:
            raise FileNotFoundError("new189_planegt 模式缺少 PlaneGT/map 和 Sandiego_gt。")
        gt = build_gt_repo_style("Sandiego", new_cube, gt_src)
        note = "Sandiego via 189-band packaged version, preferring PlaneGT/map for repo alignment"
        return DatasetPack("Sandiego", new_cube, gt, note)

    if mode == "new189_newgt":
        if new_cube is None or gt_new is None:
            raise FileNotFoundError("new189_newgt 模式需要 Sandiego_new.mat + Sandiego_gt.mat。")
        gt = build_gt_repo_style("Sandiego", new_cube, gt_new)
        note = "Sandiego via Sandiego_new.mat + Sandiego_gt.mat"
        return DatasetPack("Sandiego", new_cube, gt, note)

    if mode != "best_effort_repo":
        raise ValueError(f"Unknown sandiego mode: {mode}")

    # 默认：尽量贴近原仓库，但仅使用现在已有的数据
    if old_cube is not None and gt_repo is not None:
        # 情况 A：如果是大尺寸母场景，就尽量按原仓库 k==1 原样处理
        if old_cube.shape[0] >= 150 and old_cube.shape[1] >= 150:
            cube = remove_1based_bands(old_cube, BAD_BANDS_REPO_K1)
            cube = cube[0:150, 0:150, :]
            gt = build_gt_repo_style("Sandiego", cube, gt_repo)
            note = "Sandiego best_effort_repo: using original-style k==1 path (repo bad bands + crop [:150,:150] + PlaneGT)"
            return DatasetPack("Sandiego", cube, gt, note)

        # 情况 B：如果上传的 Sandiego.mat 已是 100x100x224 的裁剪版，无法精确还原 150x150 母场景
        if old_cube.shape[:2] == (100, 100) and old_cube.shape[2] == 224:
            if new_cube is not None and new_cube.shape[:2] == (100, 100):
                cube = new_cube
                gt = build_gt_repo_style("Sandiego", cube, gt_repo)
                note = (
                    "Sandiego best_effort_repo: uploaded Sandiego.mat is already a 100x100 cropped scene, "
                    "so exact repo k==1 150x150 reconstruction is impossible; using provided Sandiego_new.mat "
                    "+ PlaneGT/map as the closest repo-aligned pair"
                )
                return DatasetPack("Sandiego", cube, gt, note)

            cube = remove_1based_bands(old_cube, BAD_BANDS_UPLOADED_TO_NEW189)
            gt = build_gt_repo_style("Sandiego", cube, gt_repo)
            note = (
                "Sandiego best_effort_repo: uploaded Sandiego.mat is already a 100x100 cropped scene; "
                "constructed 189-band cube from Sandiego.mat using uploaded-compatible bad-band list, and used PlaneGT/map"
            )
            return DatasetPack("Sandiego", cube, gt, note)

        # 情况 C：其它旧版 Sandiego.mat，尽量用 PlaneGT / map 直接跑
        cube = old_cube
        gt = build_gt_repo_style("Sandiego", cube, gt_repo)
        note = "Sandiego best_effort_repo: using available Sandiego.mat + PlaneGT/map directly"
        return DatasetPack("Sandiego", cube, gt, note)

    if new_cube is not None and gt_repo is not None:
        gt = build_gt_repo_style("Sandiego", new_cube, gt_repo)
        note = "Sandiego best_effort_repo: using Sandiego_new.mat + PlaneGT/map (repo-aligned GT)"
        return DatasetPack("Sandiego", new_cube, gt, note)

    if new_cube is not None and gt_new is not None:
        gt = build_gt_repo_style("Sandiego", new_cube, gt_new)
        note = "Sandiego best_effort_repo fallback: using Sandiego_new.mat + Sandiego_gt.mat"
        return DatasetPack("Sandiego", new_cube, gt, note)

    raise FileNotFoundError(
        "Sandiego data not found. Expected one of: (Sandiego.mat + PlaneGT.mat), (Sandiego_new.mat + PlaneGT.mat), or (Sandiego_new.mat + Sandiego_gt.mat)."
    )


def dataset_packs(data_dir: Path, sandiego_mode: str) -> List[DatasetPack]:
    # return [prepare_urban(data_dir), prepare_sandiego(data_dir, mode=sandiego_mode)]
    return [prepare_sandiego(data_dir, mode=sandiego_mode)]


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


def roc_curve_pf_pd_masked(scores: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (np.asarray(mask) > 0).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)[mask]
    y = (np.asarray(gt).reshape(-1)[mask] > 0).astype(np.uint8)

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


def auc_on_mask(scores: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    pf, pd = roc_curve_pf_pd_masked(scores, gt, mask)
    return auc_trapz(pf, pd)


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
    local_opts = {
        "lambda_": lam,
        "lambdaVal": lam,
        "lambda_val": lam,
        "lambda": lam,
        "maxIter": 1000,
        "rho": 1.03,
        "tol": 1e-6,
    }
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


def method_score(cube_norm: np.ndarray, method: str, base_lam: float, opts: Dict) -> np.ndarray:
    if method.upper() == "SSCTV":
        return detector_SSCTV(cube_norm, base_lam, opts)
    elif method.upper() in ("SSCTV_LOGDET", "SSCTV-LOGDET", "LOGDET"):
        return detector_SSCTV_LOGDET(cube_norm, base_lam, opts)
    else:
        raise RuntimeError(f"Unknown method: {method}")


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
    auc_full = auc_trapz(pf, pd)
    save_roc(pf, pd, auc_full, f"{dataset.name} • {method_name}", out_dir / f"{base}_ROC.png")

    sec = float(time.perf_counter() - t0)
    return {
        "dataset": dataset.name,
        "method": method_name,
        "auc_full": float(auc_full),
        "time_sec": sec,
        "score_min": float(np.nanmin(score)),
        "score_max": float(np.nanmax(score)),
        "score_mean": float(np.nanmean(score)),
        "score_std": float(np.nanstd(score)),
        "lambda_used": lam,
        "notes": dataset.notes,
    }


def parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for x in str(text).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError(f"Empty float list: {text}")
    return vals


def make_tune_test_masks(
    gt: np.ndarray,
    tune_pos_frac: float = 0.5,
    tune_neg_pos_ratio: int = 20,
    tune_neg_cap: int = 20000,
    seed: int = 3407,
) -> Tuple[np.ndarray, np.ndarray]:
    y = (np.asarray(gt).reshape(-1) > 0).astype(np.uint8)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if pos_idx.size < 2:
        raise RuntimeError("Positive anomaly pixels are too few for tune/test split.")
    if neg_idx.size < 2:
        raise RuntimeError("Negative background pixels are too few for tune/test split.")

    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(pos_idx)
    neg_idx = rng.permutation(neg_idx)

    n_pos_tune = max(1, int(round(tune_pos_frac * pos_idx.size)))
    n_pos_tune = min(n_pos_tune, pos_idx.size - 1)

    desired_neg_tune = max(n_pos_tune * int(tune_neg_pos_ratio), n_pos_tune)
    n_neg_tune = min(desired_neg_tune, tune_neg_cap, neg_idx.size - 1)

    tune_idx = np.concatenate([pos_idx[:n_pos_tune], neg_idx[:n_neg_tune]])
    test_idx = np.concatenate([pos_idx[n_pos_tune:], neg_idx[n_neg_tune:]])

    tune_mask = np.zeros_like(y, dtype=np.uint8)
    test_mask = np.zeros_like(y, dtype=np.uint8)
    tune_mask[tune_idx] = 1
    test_mask[test_idx] = 1

    tune_mask = tune_mask.reshape(gt.shape)
    test_mask = test_mask.reshape(gt.shape)
    return tune_mask, test_mask


def build_ssctv_candidates(base_lam: float, args: argparse.Namespace) -> List[Dict]:
    scales = parse_float_list(args.ssctv_lambda_scales)
    out = []
    for s in scales:
        lam = float(base_lam * s)
        out.append({
            "lambda": lam,
            "lambda_": lam,
            "lambdaVal": lam,
            "lambda_val": lam,
            "maxIter": args.ssctv_maxIter,
            "rho": args.ssctv_rho,
            "tol": args.ssctv_tol,
            "_meta_lambda_scale": s,
            "_meta_lambda_abs": lam,
        })
    return out


def build_logdet_candidates(base_lam: float, args: argparse.Namespace) -> List[Dict]:
    lambda1_scales = parse_float_list(args.logdet_lambda1_scales)
    lambda2_ratios = parse_float_list(args.logdet_lambda2_ratios)
    alpha_list = parse_float_list(args.logdet_alpha_list)

    out = []
    for s1 in lambda1_scales:
        lam1 = float(base_lam * s1)
        for r2 in lambda2_ratios:
            lam2 = float(lam1 * r2)
            for a in alpha_list:
                out.append({
                    "lambda1": lam1,
                    "lambda_1": lam1,
                    "lambda2": lam2,
                    "lambda_2": lam2,
                    "alpha1": float(a),
                    "alpha2": float(a),
                    "maxIter": args.logdet_maxIter,
                    "rho": args.logdet_rho,
                    "tol": args.logdet_tol,
                    "_meta_lambda1_scale": s1,
                    "_meta_lambda1_abs": lam1,
                    "_meta_lambda2_ratio": r2,
                    "_meta_lambda2_abs": lam2,
                    "_meta_alpha": a,
                })
    return out


def strip_meta(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if not k.startswith("_meta_")}


def tune_method(
    dataset: DatasetPack,
    method: str,
    cube_norm: np.ndarray,
    base_lam: float,
    tune_mask: np.ndarray,
    candidate_opts: List[Dict],
    out_dir: Path,
) -> Tuple[Dict, List[Dict]]:
    records: List[Dict] = []
    best_auc = -np.inf
    best_opts: Optional[Dict] = None

    for i, cand in enumerate(candidate_opts, start=1):
        pure_opts = strip_meta(cand)
        score = method_score(cube_norm, method, base_lam, pure_opts)
        auc_val = auc_on_mask(score, dataset.gt, tune_mask)

        rec = {
            "dataset": dataset.name,
            "method": method,
            "trial_id": i,
            "auc_tune": float(auc_val),
        }
        for k, v in cand.items():
            if k.startswith("_meta_"):
                rec[k.replace("_meta_", "")] = v
        records.append(rec)

        print(
            f"[TUNE] {dataset.name} | {method:12s} | trial {i:03d}/{len(candidate_opts):03d} | AUC_tune={auc_val:.6f}",
            flush=True,
        )

        if auc_val > best_auc:
            best_auc = auc_val
            best_opts = pure_opts.copy()

    if best_opts is None:
        raise RuntimeError(f"Failed to select best config for {dataset.name} / {method}")

    csv_path = out_dir / f"{dataset.name}_{method}_tuning_records.csv"
    save_records(records, csv_path)
    return best_opts, records


def save_records(records: List[Dict], out_csv: Path) -> None:
    if not records:
        return
    fieldnames = []
    seen = set()
    preferred = ["dataset", "method", "trial_id", "auc_tune"]
    for k in preferred:
        if any(k in r for r in records):
            fieldnames.append(k)
            seen.add(k)
    for r in records:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def save_summary(results: List[Dict], out_dir: Path) -> None:
    json_path = out_dir / "main_ssctv_auc_summary.json"
    csv_path = out_dir / "main_ssctv_auc_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if not results:
        return

    preferred = [
        "dataset", "method",
        "auc_holdout", "auc_full", "auc_tune_best",
        "time_sec",
        "score_min", "score_max", "score_mean", "score_std",
        "lambda_used",
        "selected_lambda_scale",
        "selected_lambda_abs",
        "selected_lambda1_scale", "selected_lambda1_abs",
        "selected_lambda2_ratio", "selected_lambda2_abs",
        "selected_alpha",
        "selection_protocol",
        "notes",
    ]
    fieldnames = []
    seen = set()
    for k in preferred:
        if any(k in r for r in results):
            fieldnames.append(k)
            seen.add(k)
    for r in results:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fair comparison of SSCTV and SSCTV_LOGDET using the same validation/hold-out protocol."
    )
    p.add_argument("--out_dir", type=str, default=str(ROOT), help="Output directory. Default: script folder")
    p.add_argument(
        "--sandiego_mode",
        type=str,
        default="best_effort_repo",
        choices=["best_effort_repo", "repo_mainm_186", "new189_planegt", "new189_newgt"],
        help=(
            "Sandiego loading mode. best_effort_repo=default and recommended; "
            "repo_mainm_186=force original repo bad bands; "
            "new189_planegt=force 189-band cube with PlaneGT/map; "
            "new189_newgt=force Sandiego_new + Sandiego_gt."
        ),
    )

    # tuning split
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--tune_pos_frac", type=float, default=0.5)
    p.add_argument("--tune_neg_pos_ratio", type=int, default=20)
    p.add_argument("--tune_neg_cap", type=int, default=20000)

    # SSCTV base solver
    p.add_argument("--ssctv_maxIter", type=int, default=1000)
    p.add_argument("--ssctv_rho", type=float, default=1.03)
    p.add_argument("--ssctv_tol", type=float, default=1e-6)

    # LOGDET base solver
    p.add_argument("--logdet_maxIter", type=int, default=1000)
    p.add_argument("--logdet_rho", type=float, default=1.03)
    p.add_argument("--logdet_tol", type=float, default=1e-6)

    # fair tuning grids
    p.add_argument("--ssctv_lambda_scales", type=str, default="0.75,1.0,1.25")
    p.add_argument("--logdet_lambda1_scales", type=str, default="0.75,1.0,1.25")
    p.add_argument("--logdet_lambda2_ratios", type=str, default="0.03,0.1,0.3")
    p.add_argument("--logdet_alpha_list", type=str, default="0.5,1.0,2.0")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    packs = dataset_packs(DATA_DIR, args.sandiego_mode)
    for ds in packs:
        save_binary_map(ds.gt, out_dir / f"{ds.name}_GT_repo_style.png")

    results: List[Dict] = []

    for ds in packs:
        print(f"\n===== Dataset: {ds.name} =====", flush=True)
        print(f"[INFO] {ds.notes}", flush=True)

        cube_norm = repo_pixel_l2_normalize_cube(ds.cube)
        base_lam = repo_style_lambda(cube_norm)

        tune_mask, test_mask = make_tune_test_masks(
            ds.gt,
            tune_pos_frac=args.tune_pos_frac,
            tune_neg_pos_ratio=args.tune_neg_pos_ratio,
            tune_neg_cap=args.tune_neg_cap,
            seed=args.seed,
        )

        save_binary_map(tune_mask, out_dir / f"{ds.name}_TUNE_MASK.png")
        save_binary_map(test_mask, out_dir / f"{ds.name}_TEST_MASK.png")

        ssctv_candidates = build_ssctv_candidates(base_lam, args)
        best_ssctv_opts, ssctv_records = tune_method(
            dataset=ds,
            method="SSCTV",
            cube_norm=cube_norm,
            base_lam=base_lam,
            tune_mask=tune_mask,
            candidate_opts=ssctv_candidates,
            out_dir=out_dir,
        )

        ssctv_score = method_score(cube_norm, "SSCTV", base_lam, best_ssctv_opts)
        ssctv_auc_holdout = auc_on_mask(ssctv_score, ds.gt, test_mask)
        ssctv_auc_tune_best = max(r["auc_tune"] for r in ssctv_records)

        print(f"[BEST] {ds.name} | SSCTV        | holdout AUC={ssctv_auc_holdout:.6f}", flush=True)

        ssctv_res = run_one_method(ds, "SSCTV", out_dir, best_ssctv_opts)
        ssctv_best_meta = max(ssctv_records, key=lambda r: r["auc_tune"])
        ssctv_res.update({
            "auc_holdout": float(ssctv_auc_holdout),
            "auc_tune_best": float(ssctv_auc_tune_best),
            "selected_lambda_scale": float(ssctv_best_meta["lambda_scale"]),
            "selected_lambda_abs": float(ssctv_best_meta["lambda_abs"]),
            "selection_protocol": "same validation split, then hold-out comparison",
        })
        results.append(ssctv_res)

        logdet_candidates = build_logdet_candidates(base_lam, args)
        best_logdet_opts, logdet_records = tune_method(
            dataset=ds,
            method="SSCTV_LOGDET",
            cube_norm=cube_norm,
            base_lam=base_lam,
            tune_mask=tune_mask,
            candidate_opts=logdet_candidates,
            out_dir=out_dir,
        )

        logdet_score = method_score(cube_norm, "SSCTV_LOGDET", base_lam, best_logdet_opts)
        logdet_auc_holdout = auc_on_mask(logdet_score, ds.gt, test_mask)
        logdet_auc_tune_best = max(r["auc_tune"] for r in logdet_records)

        print(f"[BEST] {ds.name} | SSCTV_LOGDET | holdout AUC={logdet_auc_holdout:.6f}", flush=True)

        logdet_res = run_one_method(ds, "SSCTV_LOGDET", out_dir, best_logdet_opts)
        logdet_best_meta = max(logdet_records, key=lambda r: r["auc_tune"])
        logdet_res.update({
            "auc_holdout": float(logdet_auc_holdout),
            "auc_tune_best": float(logdet_auc_tune_best),
            "selected_lambda1_scale": float(logdet_best_meta["lambda1_scale"]),
            "selected_lambda1_abs": float(logdet_best_meta["lambda1_abs"]),
            "selected_lambda2_ratio": float(logdet_best_meta["lambda2_ratio"]),
            "selected_lambda2_abs": float(logdet_best_meta["lambda2_abs"]),
            "selected_alpha": float(logdet_best_meta["alpha"]),
            "selection_protocol": "same validation split, then hold-out comparison",
        })
        results.append(logdet_res)

    save_summary(results, out_dir)

    print("\n========== FAIR COMPARISON SUMMARY ==========")
    for ds_name in ("Urban", "Sandiego"):
        ds_rows = [x for x in results if x["dataset"] == ds_name]
        if not ds_rows:
            continue
        for r in ds_rows:
            print(
                f"{r['dataset']:10s} | {r['method']:12s} | "
                f"AUC_holdout={r['auc_holdout']:.6f} | AUC_full={r['auc_full']:.6f} | "
                f"time={r['time_sec']:.2f}s"
            )

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()