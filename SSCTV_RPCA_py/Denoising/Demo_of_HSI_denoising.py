# Demo_of_HSI_denoising.py
# -*- coding: utf-8 -*-
"""
更新点：
1) 默认：使用所有CPU核并行跑“所有数据集 + 所有case”
   - 除非你在终端显式输入 --case 或 --jobs/--job 才覆盖默认
2) 输出整理到 result/ 文件夹（每次运行一个时间戳子目录）
   - 指标CSV、平均结果CSV、每个case的图片都保存到 result/xxx/...
3) 数据集扩展更容易：BaseDataset + registry（MAT / CAVE）
4) MAT 修正：默认从 Denoising/data/ 目录读取所有 pure_*.mat（多文件）
   - 也兼容你手动指定单个 .mat 文件
   - CAVE 部分不改动
"""

import os as _os

# （可选）在 import numpy 前限制 BLAS 线程，避免多进程时“过度并行”
# 你可以通过环境变量控制：HISIDEMO_NUM_THREADS=1
_NUM_THREADS = _os.environ.get("HISIDEMO_NUM_THREADS", "").strip()
if _NUM_THREADS:
    for _var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        _os.environ[_var] = _NUM_THREADS

import os
import re
import time
import glob
import sys
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# 保存图片用无GUI后端，避免跑ALL时弹窗、也更适合多进程
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# image reader (png/bmp)
try:
    import imageio.v2 as imageio
except Exception:
    import imageio

# 让 methods/ 可被导入（与原脚本一致）
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from methods.ssctv_rpca import ssctv_rpca
    HAS_SSCTV = True
except Exception:
    HAS_SSCTV = False

from methods.ctv_rpca import ctv_rpca
from methods.inexact_alm_rpca import inexact_alm_rpca


# =========================
# 工具函数
# =========================
def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def Normalize(cube: np.ndarray) -> np.ndarray:
    """
    按波段 min-max 到 [0,1]
    """
    M, N, p = cube.shape
    out = np.empty_like(cube, dtype=np.float64)
    for k in range(p):
        band = cube[:, :, k].astype(np.float64)
        vmin, vmax = band.min(), band.max()
        out[:, :, k] = (band - vmin) / (vmax - vmin + 1e-12)
    return out


def GetNoise(clean: np.ndarray,
             gaussian_level: float = 0.0,
             sparse_level: float = 0.3,
             stripe_prob: float = 0.0,
             stripe_strength: float = 0.1) -> np.ndarray:
    """
    生成有噪数据：高斯 + 稀疏脉冲 + (可选)条纹
    """
    M, N, p = clean.shape

    g = gaussian_level * np.random.randn(M, N, p)

    sp_mask = (np.random.rand(M, N, p) < sparse_level).astype(np.float64)
    sp = sp_mask * (2 * np.random.rand(M, N, p) - 1.0)  # [-1,1]

    stripe = np.zeros_like(clean)
    if stripe_prob > 0:
        choose = np.random.rand(N) < stripe_prob
        for j, flag in enumerate(choose):
            if flag:
                stripe[:, j, :] = stripe_strength * (2 * np.random.rand(1, 1, p) - 1.0)

    noise = g + sp + stripe
    noisy = np.clip(clean + noise, 0.0, 1.0)
    return noisy


def msqia(clean: np.ndarray, rec: np.ndarray):
    """
    计算 mPSNR / mSSIM / ERGAS（逐波段平均）
    GT（Ground Truth）就是 clean（加噪前的干净真值）
    """
    M, N, p = clean.shape
    psnrs, ssims = [], []
    for k in range(p):
        gt = clean[:, :, k]                 # GT
        im = np.clip(rec[:, :, k], 0, 1)
        psnrs.append(psnr(gt, im, data_range=1.0))
        try:
            ssims.append(ssim(gt, im, data_range=1.0))
        except Exception:
            pass

    mpsnr = float(np.mean(psnrs))
    mssim = float(np.mean(ssims) if ssims else np.nan)

    clean2 = clean.reshape(-1, p)
    rec2 = np.clip(rec, 0, 1).reshape(-1, p)
    rmse = np.sqrt(np.mean((clean2 - rec2) ** 2, axis=0))
    mean_ref = np.maximum(np.mean(clean2, axis=0), 1e-12)
    ergas = 100.0 * np.sqrt(np.mean((rmse / mean_ref) ** 2))
    return mpsnr, mssim, float(ergas)


def choose_rgb_bands_0based(p: int, prefer_1based=(5, 95, 125)) -> List[int]:
    """
    可视化选择 3 个波段（0基索引）
    - p>=125 用 (5,95,125)（更贴近论文/Matlab写法，1基）
    - p较小（如 CAVE 31 band）用 20%/50%/80% 分位波段
    """
    if p <= 0:
        return [0, 0, 0]

    if p >= max(prefer_1based):
        idx = [x - 1 for x in prefer_1based]
    else:
        b1 = max(1, int(round(p * 0.20)))
        b2 = max(1, int(round(p * 0.50)))
        b3 = max(1, int(round(p * 0.80)))
        idx = [b1 - 1, b2 - 1, b3 - 1]

    idx = [int(np.clip(i, 0, p - 1)) for i in idx]
    return idx


def _safe_case_dir_for_mat(sample_id: str) -> str:
    """
    MAT 多文件并行时，sample_id 可能是 'C:\\...\\pure_xxx.mat'，不能直接当目录名。
    这里用 basename（含扩展名也行）并做字符清洗。
    """
    base = os.path.basename(sample_id)
    # 只保留常见安全字符
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", base)
    return base


# =========================
# CAVE 数据集读取（保持不改动）
# =========================
def _read_gray_image_as_float(path: str) -> np.ndarray:
    """
    读取单通道图像，返回 float64 (H,W)
    - 若读出为 RGB/多通道，则取第 0 通道
    """
    img = imageio.imread(path)
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float64)


def list_cave_cases(cave_root: str) -> List[str]:
    if not os.path.isdir(cave_root):
        return []
    cases = []
    for name in os.listdir(cave_root):
        p = os.path.join(cave_root, name)
        if os.path.isdir(p):
            cases.append(name)
    cases.sort()
    return cases


def load_cave_cube(cave_root: str, case_name: str) -> np.ndarray:
    """
    兼容：
      data/CAVE/balloons_ms/balloons_ms/balloons_ms_01.png
      ...
      data/CAVE/balloons_ms/balloons_ms/balloons_ms_31.png
    """
    folder_a = os.path.join(cave_root, case_name, case_name)  # 两层
    folder_b = os.path.join(cave_root, case_name)             # 一层

    if os.path.isdir(folder_a):
        folder = folder_a
    elif os.path.isdir(folder_b):
        folder = folder_b
    else:
        raise FileNotFoundError(f"CAVE 案例目录不存在：{folder_a} 或 {folder_b}")

    if case_name.endswith("_ms"):
        prefix = case_name
    else:
        prefix = f"{case_name}_ms"

    cand = glob.glob(os.path.join(folder, f"{prefix}_*.png"))
    if not cand:
        cand = glob.glob(os.path.join(folder, "*_ms_*.png"))
    if not cand:
        raise FileNotFoundError(
            f"在 {folder} 下未找到波段 png（期望类似 {prefix}_01.png / {prefix}_ms_01.png）"
        )

    def _extract_idx(fp: str) -> int:
        base = os.path.basename(fp)
        m = re.search(r"(\d+)\.png$", base)
        return int(m.group(1)) if m else 10**9

    cand = sorted(cand, key=_extract_idx)

    bands = []
    for fp in cand:
        bands.append(_read_gray_image_as_float(fp))

    cube = np.stack(bands, axis=-1)  # (H,W,B)
    return cube


# =========================
# 数据集扩展：BaseDataset + registry
# =========================
class BaseDataset:
    name: str = "BASE"

    def list_samples(self) -> List[str]:
        raise NotImplementedError

    def load(self, sample_id: str) -> Tuple[np.ndarray, str]:
        raise NotImplementedError


@dataclass
class MatDataset(BaseDataset):
    """
    MAT 修正：
    - mat_path 可以是：单个 .mat 文件路径；或一个目录路径
    - 如果是目录：自动读取其中所有 pure_*.mat
    """
    name: str = "MAT"
    mat_path: str = ""
    mat_var: str = "Ori_H"

    def list_samples(self) -> List[str]:
        p = self.mat_path

        # 1) 指定单个文件
        if os.path.isfile(p) and p.lower().endswith(".mat"):
            return [p]

        # 2) 指定目录：读取 pure_*.mat
        if os.path.isdir(p):
            mats = glob.glob(os.path.join(p, "pure_*.mat"))
            mats.sort()
            return mats

        # 3) 既不是文件也不是目录：尝试当作 glob pattern
        mats = glob.glob(p)
        mats = [x for x in mats if x.lower().endswith(".mat")]
        mats.sort()
        return mats

    def load(self, sample_id: str) -> Tuple[np.ndarray, str]:
        mat_file = sample_id
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"找不到 .mat：{mat_file}")

        md = loadmat(mat_file)
        if self.mat_var not in md:
            raise KeyError(f"在 {mat_file} 里找不到变量 {self.mat_var}")

        clean_data = np.asarray(md[self.mat_var], dtype=np.float64)
        clean_data = Normalize(clean_data)
        case_id = os.path.basename(mat_file)
        return clean_data, case_id


@dataclass
class CaveDataset(BaseDataset):
    name: str = "CAVE"
    cave_root: str = ""

    def list_samples(self) -> List[str]:
        return list_cave_cases(self.cave_root)

    def load(self, sample_id: str) -> Tuple[np.ndarray, str]:
        clean_data = load_cave_cube(self.cave_root, sample_id)
        clean_data = Normalize(clean_data)
        return clean_data, sample_id


def build_dataset(name: str, args) -> BaseDataset:
    name = name.upper().strip()
    if name == "MAT":
        return MatDataset(mat_path=args.mat_path, mat_var=args.mat_var)
    if name == "CAVE":
        return CaveDataset(cave_root=args.cave_root)
    raise ValueError(f"未知数据集：{name}（目前支持 MAT / CAVE / ALL 由主流程控制）")


# =========================
# 保存结果：CSV + 图片
# =========================
def save_results_csv(path: str, all_case_results: List[Tuple[str, list]]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "method", "mPSNR", "mSSIM", "ERGAS", "time_sec"])
        for case_id, res in all_case_results:
            for (name, mpsnr_, mssim_, ergas_, sec) in res:
                w.writerow([case_id, name, f"{mpsnr_:.6f}", f"{mssim_:.6f}", f"{ergas_:.6f}", f"{sec:.6f}"])


def aggregate_all_results(all_case_results: List[Tuple[str, list]]) -> Dict[str, np.ndarray]:
    bucket: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for case_id, res in all_case_results:
        for (name, mpsnr_, mssim_, ergas_, sec) in res:
            bucket.setdefault(name, []).append((mpsnr_, mssim_, ergas_, sec))

    out = {}
    for name, vals in bucket.items():
        out[name] = np.array(vals, dtype=np.float64)
    return out


def save_average_csv(path: str, agg: Dict[str, np.ndarray]):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "AVG_mPSNR", "AVG_mSSIM", "AVG_ERGAS", "AVG_time_sec", "N_cases"])
        for name, arr in agg.items():
            w.writerow([
                name,
                f"{np.mean(arr[:, 0]):.6f}",
                f"{np.mean(arr[:, 1]):.6f}",
                f"{np.mean(arr[:, 2]):.6f}",
                f"{np.mean(arr[:, 3]):.6f}",
                int(arr.shape[0]),
            ])


# =========================
# 核心：单case运行 + 保存图片
# =========================
def run_one_case(clean_data: np.ndarray,
                 case_id: str,
                 gaussian_level: float,
                 sparse_level: float,
                 stripe_prob: float,
                 stripe_strength: float,
                 seed: int,
                 out_case_dir: str,
                 save_images: bool = True):
    ensure_dir(out_case_dir)
    img_dir = ensure_dir(os.path.join(out_case_dir, "images"))

    if seed is not None:
        np.random.seed(int(seed))

    M, N, p = clean_data.shape

    noise_data = GetNoise(clean_data,
                          gaussian_level=gaussian_level,
                          sparse_level=sparse_level,
                          stripe_prob=stripe_prob,
                          stripe_strength=stripe_strength)

    results = []
    cubes = {"Clean": clean_data, "Noisy": noise_data}

    mpsnr0, mssim0, ergas0 = msqia(clean_data, noise_data)
    results.append(("Noisy", mpsnr0, mssim0, ergas0, 0.0))

    # SSCTV-RPCA
    if HAS_SSCTV:
        opts = dict(rho=1.1, lambda_=2.0 / np.sqrt(M * N), maxIter=200, tol=1e-6)
        t0 = time.time()
        X_ssctv, _ = ssctv_rpca(noise_data, opts)
        tcost = time.time() - t0
        r = msqia(clean_data, X_ssctv)
        results.append(("SSCTV-RPCA",) + r + (tcost,))
        cubes["SSCTV-RPCA"] = X_ssctv
        cubes["SSCTV-Error"] = np.abs(clean_data - np.clip(X_ssctv, 0, 1))

    # CTV-RPCA
    try:
        opts = dict(rho=1.5, lambda_=3.0 / np.sqrt(M * N), maxIter=200, tol=1e-6)
        t0 = time.time()
        X_ctv, _ = ctv_rpca(noise_data, opts)
        tcost = time.time() - t0
        r = msqia(clean_data, X_ctv)
        results.append(("CTV-RPCA",) + r + (tcost,))
        cubes["CTV-RPCA"] = X_ctv
        cubes["CTV-Error"] = np.abs(clean_data - np.clip(X_ctv, 0, 1))
    except Exception:
        pass

    # RPCA（IALM）
    D = noise_data.reshape(M * N, p, order="F")
    t0 = time.time()
    L_hat, _, _ = inexact_alm_rpca(
        D,
        lambda_=1.0 / np.sqrt(max(M * N, p)),
        maxIter=1000,
        tol=1e-6,
        verbose=False
    )
    tcost = time.time() - t0
    X_rpca = L_hat.reshape(M, N, p, order="F")
    r = msqia(clean_data, X_rpca)
    results.append(("RPCA",) + r + (tcost,))
    cubes["RPCA"] = X_rpca
    cubes["RPCA-Error"] = np.abs(clean_data - np.clip(X_rpca, 0, 1))

    if save_images:
        rgb_b = choose_rgb_bands_0based(p)
        b_show = [i + 1 for i in rgb_b]

        def _to_rgb(cube: np.ndarray, idx3):
            r_ = cube[:, :, idx3[0]]
            g_ = cube[:, :, idx3[1]]
            b_ = cube[:, :, idx3[2]]
            rgb = np.stack([r_, g_, b_], axis=-1)
            return np.clip(rgb, 0, 1)

        show_order = ["Clean", "Noisy", "SSCTV-RPCA", "CTV-RPCA", "RPCA"]
        mid_band = int(np.clip(round(p * 0.5) - 1, 0, p - 1))

        plot_items = [k for k in show_order if k in cubes]
        if "SSCTV-Error" in cubes:
            plot_items.append("SSCTV-Error")
        elif "CTV-Error" in cubes:
            plot_items.append("CTV-Error")
        elif "RPCA-Error" in cubes:
            plot_items.append("RPCA-Error")

        cols = 3
        rows = int(np.ceil(len(plot_items) / cols))
        plt.figure(figsize=(16, 10))

        for i, name in enumerate(plot_items, start=1):
            plt.subplot(rows, cols, i)
            if name.endswith("-Error"):
                em = cubes[name][:, :, mid_band]
                plt.imshow(em, cmap="gray")
                plt.title(f"{name} (band {mid_band+1})")
            else:
                plt.imshow(_to_rgb(cubes[name], rgb_b))
                plt.title(f"{name} (RGB {b_show[0]}-{b_show[1]}-{b_show[2]})")
            plt.axis("off")

        plt.suptitle(f"Denoising - {case_id}", y=0.98, fontsize=14)
        plt.tight_layout()

        out_png = os.path.join(img_dir, "overview.png")
        plt.savefig(out_png, dpi=200)
        plt.close()

    return results


# =========================
# 多进程 worker
# =========================
def _worker_run_one(payload: Dict):
    dataset_name = payload["dataset_name"]
    sample_id = payload["sample_id"]

    class _ArgsObj:
        pass

    a = _ArgsObj()
    a.mat_path = payload.get("mat_path", "")
    a.mat_var = payload.get("mat_var", "Ori_H")
    a.cave_root = payload.get("cave_root", "")

    dataset = build_dataset(dataset_name, a)
    clean_data, case_id = dataset.load(sample_id)

    res = run_one_case(
        clean_data=clean_data,
        case_id=case_id,
        gaussian_level=payload["gaussian_level"],
        sparse_level=payload["sparse_level"],
        stripe_prob=payload["stripe_prob"],
        stripe_strength=payload["stripe_strength"],
        seed=payload["seed"],
        out_case_dir=payload["out_case_dir"],
        save_images=True
    )

    return case_id, res


# =========================
# CLI：默认不传参也会跑ALL+全核
# =========================
def parse_args():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CAVE_ROOT = os.path.join(THIS_DIR, "data", "CAVE")
    DEFAULT_RESULT_ROOT = os.path.join(THIS_DIR, "result")

    # ✅ 按你说的：MAT 干净数据统一在 Denoising/data 下，并且 pure_ 开头
    DEFAULT_MAT_DIR = r"C:\Users\24455\PycharmProjects\PythonProject2\SSCTV_RPCA_py\Denoising\data"

    p = argparse.ArgumentParser(description="HSI/MSI Denoising Demo (default: ALL datasets + ALL cases + all CPU cores)")

    p.add_argument("--dataset", type=str, default="ALL", choices=["ALL", "MAT", "CAVE"],
                   help="默认 ALL：跑 MAT + CAVE。也可指定 MAT 或 CAVE")

    p.add_argument("--case", type=str, default="ALL",
                   help="CAVE 样本ID（如 balloons_ms / flowers_ms / ALL）。默认 ALL。MAT 会忽略该参数。")

    p.add_argument("--jobs", type=int, default=None,
                   help="并行进程数（默认：如果跑ALL则使用CPU所有核）。")
    p.add_argument("--job", type=int, default=None,
                   help="同 --jobs（兼容参数名）。")

    p.add_argument("--gaussian", type=float, default=0.0)
    p.add_argument("--sparse", type=float, default=0.3)
    p.add_argument("--stripe_prob", type=float, default=0.0)
    p.add_argument("--stripe_strength", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=0)

    # ✅ mat_path 现在默认是“目录”，脚本会自动扫描其中 pure_*.mat
    p.add_argument("--mat_path", type=str, default=DEFAULT_MAT_DIR,
                   help="MAT：可指定单个.mat文件或目录；若为目录则读取其中 pure_*.mat")
    p.add_argument("--mat_var", type=str, default="Ori_H")

    p.add_argument("--cave_root", type=str, default=DEFAULT_CAVE_ROOT)

    p.add_argument("--result_root", type=str, default=DEFAULT_RESULT_ROOT,
                   help="结果输出根目录，默认 result/")

    return p.parse_args()


# =========================
# 主控制：默认全核跑ALL数据集
# =========================
def run_dataset_cases(dataset_name: str,
                      args,
                      run_dir: str,
                      case: str,
                      jobs: int,
                      gaussian_level: float,
                      sparse_level: float,
                      stripe_prob: float,
                      stripe_strength: float,
                      base_seed: int):
    ds_dir = ensure_dir(os.path.join(run_dir, dataset_name.upper()))
    dataset = build_dataset(dataset_name, args)

    if dataset_name.upper() == "MAT":
        samples = dataset.list_samples()
    else:
        if case.upper() == "ALL":
            samples = dataset.list_samples()
        else:
            samples = [case]

    if not samples:
        raise FileNotFoundError(f"{dataset_name} 未找到任何样本，请检查路径。")

    all_case_results: List[Tuple[str, list]] = []

    use_parallel = (len(samples) > 1 and jobs > 1)

    if not use_parallel:
        for idx, sample_id in enumerate(samples, start=1):
            cur_seed = base_seed + idx
            clean_data, case_id = dataset.load(sample_id)

            out_case_dir = ensure_dir(os.path.join(ds_dir, case_id))
            res = run_one_case(
                clean_data=clean_data,
                case_id=case_id,
                gaussian_level=gaussian_level,
                sparse_level=sparse_level,
                stripe_prob=stripe_prob,
                stripe_strength=stripe_strength,
                seed=cur_seed,
                out_case_dir=out_case_dir,
                save_images=True
            )
            all_case_results.append((case_id, res))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = []
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            for idx, sample_id in enumerate(samples, start=1):
                cur_seed = base_seed + idx

                # ✅ 只对 MAT 做安全目录名处理；CAVE 不动
                if dataset_name.upper() == "MAT":
                    out_dir_name = _safe_case_dir_for_mat(sample_id)
                else:
                    out_dir_name = str(sample_id)

                out_case_dir = ensure_dir(os.path.join(ds_dir, out_dir_name))

                payload = dict(
                    dataset_name=dataset_name.upper(),
                    sample_id=sample_id,
                    seed=cur_seed,
                    gaussian_level=gaussian_level,
                    sparse_level=sparse_level,
                    stripe_prob=stripe_prob,
                    stripe_strength=stripe_strength,
                    out_case_dir=out_case_dir,
                    mat_path=args.mat_path,
                    mat_var=args.mat_var,
                    cave_root=args.cave_root,
                )
                futures.append(ex.submit(_worker_run_one, payload))

            for fut in as_completed(futures):
                case_id, res = fut.result()
                all_case_results.append((case_id, res))

        all_case_results.sort(key=lambda x: x[0])

    per_case_csv = os.path.join(ds_dir, "metrics_per_case.csv")
    save_results_csv(per_case_csv, all_case_results)

    agg = aggregate_all_results(all_case_results)
    avg_csv = os.path.join(ds_dir, "metrics_average.csv")
    save_average_csv(avg_csv, agg)

    return {
        "dataset": dataset_name.upper(),
        "num_cases": len(all_case_results),
        "per_case_csv": per_case_csv,
        "avg_csv": avg_csv,
    }


def run():
    args = parse_args()

    user_jobs = args.jobs if args.jobs is not None else args.job
    user_provided_jobs = (user_jobs is not None)

    case = args.case.strip()

    if user_provided_jobs:
        jobs = int(user_jobs)
    else:
        jobs = int(os.cpu_count() or 1)

    gaussian_level = float(args.gaussian)
    sparse_level = float(args.sparse)
    stripe_prob = float(args.stripe_prob)
    stripe_strength = float(args.stripe_strength)
    seed = int(args.seed)

    run_id = now_run_id()
    run_dir = ensure_dir(os.path.join(args.result_root, run_id))

    with open(os.path.join(run_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("该目录为本次运行输出。\n")
        f.write(f"run_id={run_id}\n")
        f.write(f"dataset={args.dataset}\n")
        f.write(f"case={args.case}\n")
        f.write(f"jobs={jobs} (user_provided={user_provided_jobs})\n")
        f.write(f"noise: gaussian={gaussian_level}, sparse={sparse_level}, stripe_prob={stripe_prob}, stripe_strength={stripe_strength}\n")
        f.write(f"seed={seed}\n")
        f.write(f"mat_path={args.mat_path}\n")
        f.write(f"mat_var={args.mat_var}\n")
        f.write(f"cave_root={args.cave_root}\n")

    summary_rows = []

    dataset_choice = args.dataset.upper().strip()
    if dataset_choice == "ALL":
        datasets_to_run = ["CAVE", "MAT"]
    else:
        datasets_to_run = [dataset_choice]

    for ds in datasets_to_run:
        try:
            info = run_dataset_cases(
                dataset_name=ds,
                args=args,
                run_dir=run_dir,
                case=case,
                jobs=jobs,
                gaussian_level=gaussian_level,
                sparse_level=sparse_level,
                stripe_prob=stripe_prob,
                stripe_strength=stripe_strength,
                base_seed=seed
            )
            summary_rows.append(info)
        except Exception as e:
            err_path = os.path.join(run_dir, f"{ds}_ERROR.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(str(e))
            summary_rows.append({
                "dataset": ds,
                "num_cases": 0,
                "per_case_csv": "",
                "avg_csv": "",
                "error": err_path
            })

    summary_csv = os.path.join(run_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "num_cases", "per_case_csv", "avg_csv", "error_file"])
        for row in summary_rows:
            w.writerow([
                row.get("dataset", ""),
                row.get("num_cases", 0),
                row.get("per_case_csv", ""),
                row.get("avg_csv", ""),
                row.get("error", "")
            ])

    print(f"\n[Done] 本次结果已输出到：{run_dir}")
    print(f"[Done] 总览：{summary_csv}")
    for row in summary_rows:
        ds = row.get("dataset", "")
        if row.get("num_cases", 0) > 0:
            print(f"  - {ds}: {row['num_cases']} cases")
            print(f"    per-case: {row['per_case_csv']}")
            print(f"    average : {row['avg_csv']}")
        else:
            print(f"  - {ds}: FAILED or empty. error={row.get('error','')}")


if __name__ == "__main__":
    run()
