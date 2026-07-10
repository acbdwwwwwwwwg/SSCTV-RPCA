from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import test2 as T


# =========================
# Fixed experiment settings
# =========================

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ablation2_single_pair_outputs"

# Strictly follow test2.py Sandiego handling
SANDIEGO_MODE = "best_effort_repo"

# Strictly follow test2.py tuning split defaults
SEED = 3407
TUNE_POS_FRAC = 0.5
TUNE_NEG_POS_RATIO = 20
TUNE_NEG_CAP = 20000

# Strictly follow test2.py solver defaults
SOLVER_DEFAULTS = {
    "maxIter": 1000,
    "rho": 1.03,
    "tol": 1e-6,
}

# Optional reference summary produced by test2.py
SUMMARY_JSON = ROOT / "main_ssctv_auc_summary.json"
DEFAULT_BEST = {
    "selected_lambda_scale": 1.0,
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_logsparsity_best() -> Dict:
    if SUMMARY_JSON.exists():
        try:
            data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for row in data:
                    if str(row.get("method", "")).upper() == "SSCTV_LOGSPARSITY":
                        best = {
                            "selected_lambda_scale": float(row.get("selected_lambda_scale", 1.0)),
                        }
                        if "selected_lambda_abs" in row:
                            best["selected_lambda_abs"] = float(row["selected_lambda_abs"])
                        if "lambda_used" in row:
                            best["lambda_used"] = float(row["lambda_used"])
                        best["source"] = str(SUMMARY_JSON)
                        return best
        except Exception:
            pass

    best = dict(DEFAULT_BEST)
    best["source"] = "fallback_default"
    return best


def get_logsparsity_solver():
    solver = getattr(T, "ssctv_log_sparsity", None)
    if solver is not None:
        return solver

    # Fallback import, in case test2.py import failed for some reason
    try:
        from methods.ssctv_log_sparsity import ssctv_log_sparsity
        return ssctv_log_sparsity
    except Exception as e:
        raise RuntimeError(
            "Cannot import methods.ssctv_log_sparsity.ssctv_log_sparsity"
        ) from e


def get_score_from_residual():
    fn = getattr(T, "_score_from_residual", None)
    if fn is None:
        raise RuntimeError("test2.py does not expose _score_from_residual")
    return fn


def load_sandiego_pack():
    packs = T.dataset_packs(T.DATA_DIR, sandiego_mode=SANDIEGO_MODE)
    sandiego = [ds for ds in packs if ds.name.lower() == "sandiego"]
    if not sandiego:
        raise RuntimeError(
            "No Sandiego dataset found via test2.py dataset_packs(DATA_DIR, 'best_effort_repo')."
        )
    if len(sandiego) > 1:
        raise RuntimeError(f"Expected exactly one Sandiego pack, got {len(sandiego)}")
    return sandiego[0]


def build_opts(base_lam: float, best_cfg: Dict, use_logsparsity: bool) -> dict:
    lam_scale = float(best_cfg.get("selected_lambda_scale", 1.0))
    lam = float(base_lam * lam_scale) if use_logsparsity else 0.0

    opts = {
        "lambda": lam,
        "lambda_": lam,
        "lambdaVal": lam,
        "lambda_val": lam,
    }
    opts.update(SOLVER_DEFAULTS)
    return opts


def normalize_score_for_vis(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    smin = float(np.nanmin(score))
    smax = float(np.nanmax(score))
    if smax > smin:
        vis = (score - smin) / (smax - smin)
    else:
        vis = np.zeros_like(score)
    vis[~np.isfinite(vis)] = 0.0
    return vis


def run_logsparsity_once(cube_norm: np.ndarray, opts: dict):
    solver = get_logsparsity_solver()
    score_from_residual = get_score_from_residual()

    t0 = time.perf_counter()
    try:
        X, S = solver(cube_norm, opts=opts)
    except TypeError:
        X, S = solver(cube_norm, **opts)
    elapsed = float(time.perf_counter() - t0)

    X = np.asarray(X, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    score = np.asarray(score_from_residual(S, cube_norm), dtype=np.float64)

    return X, S, score, elapsed


def save_branch_outputs(
    ds,
    branch_name: str,
    score: np.ndarray,
    S: np.ndarray,
    auc_full: float,
    out_dir: Path,
) -> None:
    base = f"{ds.name}_{branch_name}"

    np.save(out_dir / f"{base}_score.npy", score.astype(np.float32))
    np.save(out_dir / f"{base}_S.npy", S.astype(np.float32))

    score_vis = normalize_score_for_vis(score)
    T.save_heatmap(score_vis, out_dir / f"{base}_score.png")

    pf, pd = T.roc_curve_pf_pd(score, ds.gt)
    T.save_roc(pf, pd, auc_full, f"{ds.name} • {branch_name}", out_dir / f"{base}_ROC.png")


def summarize_branch(
    ds,
    branch_name: str,
    opts: dict,
    score: np.ndarray,
    S: np.ndarray,
    elapsed: float,
    tune_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict:
    pf, pd = T.roc_curve_pf_pd(score, ds.gt)
    auc_full = float(T.auc_trapz(pf, pd))
    auc_tune = float(T.auc_on_mask(score, ds.gt, tune_mask))
    auc_holdout = float(T.auc_on_mask(score, ds.gt, test_mask))

    s_abs = np.abs(S)
    nz_ratio = float(np.mean(s_abs > 1e-8))

    return {
        "dataset": ds.name,
        "branch": branch_name,
        "auc_full": auc_full,
        "auc_tune": auc_tune,
        "auc_holdout": auc_holdout,
        "time_sec": float(elapsed),
        "score_min": float(np.nanmin(score)),
        "score_max": float(np.nanmax(score)),
        "score_mean": float(np.nanmean(score)),
        "score_std": float(np.nanstd(score)),
        "S_fro": float(np.linalg.norm(S, ord="fro")),
        "S_l1": float(np.sum(s_abs)),
        "S_nonzero_ratio": nz_ratio,
        "lambda": float(opts["lambda"]),
        "notes": ds.notes,
    }


def save_summary_json_csv(summary: dict, out_dir: Path) -> None:
    json_path = out_dir / "ablation2_summary.json"
    csv_path = out_dir / "ablation2_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = [
        summary["with_logsparsity"],
        summary["without_logsparsity"],
    ]
    fieldnames = [
        "dataset",
        "branch",
        "auc_full",
        "auc_tune",
        "auc_holdout",
        "time_sec",
        "score_min",
        "score_max",
        "score_mean",
        "score_std",
        "S_fro",
        "S_l1",
        "S_nonzero_ratio",
        "lambda",
        "notes",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ensure_dir(OUT_DIR)

    ds = load_sandiego_pack()
    best_cfg = load_logsparsity_best()

    print("===== Single-pair ablation on Sandiego =====")
    print(f"[INFO] dataset      : {ds.name}")
    print(f"[INFO] notes        : {ds.notes}")
    print(f"[INFO] output dir   : {OUT_DIR}")
    print(f"[INFO] split seed   : {SEED}")
    print(f"[INFO] tune_pos_frac: {TUNE_POS_FRAC}")
    print(f"[INFO] neg/pos ratio: {TUNE_NEG_POS_RATIO}")
    print(f"[INFO] neg cap      : {TUNE_NEG_CAP}")
    print(f"[INFO] best config source: {best_cfg['source']}")

    # Strictly follow test2.py preprocessing
    cube_norm = T.repo_pixel_l2_normalize_cube(ds.cube)
    base_lam = float(T.repo_style_lambda(cube_norm))

    print(f"[INFO] base lambda computed from test2.py pipeline : {base_lam:.15f}")
    if "lambda_used" in best_cfg:
        print(f"[INFO] summary lambda_used                         : {best_cfg['lambda_used']:.15f}")
        print(f"[INFO] abs diff                                    : {abs(base_lam - best_cfg['lambda_used']):.3e}")
    print(f"[INFO] selected_lambda_scale                       : {float(best_cfg.get('selected_lambda_scale', 1.0)):.15f}")

    tune_mask, test_mask = T.make_tune_test_masks(
        ds.gt,
        tune_pos_frac=TUNE_POS_FRAC,
        tune_neg_pos_ratio=TUNE_NEG_POS_RATIO,
        tune_neg_cap=TUNE_NEG_CAP,
        seed=SEED,
    )

    # Save GT and masks exactly as test2.py-style binary maps
    T.save_binary_map(ds.gt, OUT_DIR / f"{ds.name}_GT_repo_style.png")
    T.save_binary_map(tune_mask, OUT_DIR / f"{ds.name}_TUNE_MASK.png")
    T.save_binary_map(test_mask, OUT_DIR / f"{ds.name}_TEST_MASK.png")

    # Branch A: with log-sparsity term
    opts_with = build_opts(base_lam, best_cfg, use_logsparsity=True)
    print("\n[RUN ] with_logsparsity")
    print(json.dumps(opts_with, indent=2))
    X_with, S_with, score_with, time_with = run_logsparsity_once(cube_norm, opts_with)
    res_with = summarize_branch(
        ds=ds,
        branch_name="SSCTV_LOGSPARSITY_with_logsparsity",
        opts=opts_with,
        score=score_with,
        S=S_with,
        elapsed=time_with,
        tune_mask=tune_mask,
        test_mask=test_mask,
    )
    save_branch_outputs(
        ds=ds,
        branch_name="SSCTV_LOGSPARSITY_with_logsparsity",
        score=score_with,
        S=S_with,
        auc_full=res_with["auc_full"],
        out_dir=OUT_DIR,
    )

    # Branch B: no log-sparsity term (ablation via lambda = 0)
    opts_without = build_opts(base_lam, best_cfg, use_logsparsity=False)
    print("\n[RUN ] without_logsparsity")
    print(json.dumps(opts_without, indent=2))
    X_without, S_without, score_without, time_without = run_logsparsity_once(cube_norm, opts_without)
    res_without = summarize_branch(
        ds=ds,
        branch_name="SSCTV_LOGSPARSITY_without_logsparsity",
        opts=opts_without,
        score=score_without,
        S=S_without,
        elapsed=time_without,
        tune_mask=tune_mask,
        test_mask=test_mask,
    )
    save_branch_outputs(
        ds=ds,
        branch_name="SSCTV_LOGSPARSITY_without_logsparsity",
        score=score_without,
        S=S_without,
        auc_full=res_without["auc_full"],
        out_dir=OUT_DIR,
    )

    delta_holdout = float(res_with["auc_holdout"] - res_without["auc_holdout"])
    delta_full = float(res_with["auc_full"] - res_without["auc_full"])
    delta_tune = float(res_with["auc_tune"] - res_without["auc_tune"])

    if delta_holdout > 0:
        verdict = "with_logsparsity_better_on_holdout"
    elif delta_holdout < 0:
        verdict = "without_logsparsity_better_on_holdout"
    else:
        verdict = "tie_on_holdout"

    summary = {
        "experiment": "single_pair_ablation_logsparsity_onoff",
        "dataset": ds.name,
        "sandiego_mode": SANDIEGO_MODE,
        "selection_protocol_reference": "match test2.py split defaults; use test2 summary-selected lambda scale when available",
        "seed": SEED,
        "tune_pos_frac": TUNE_POS_FRAC,
        "tune_neg_pos_ratio": TUNE_NEG_POS_RATIO,
        "tune_neg_cap": TUNE_NEG_CAP,
        "base_lambda_from_test2_pipeline": base_lam,
        "best_config_reference": best_cfg,
        "with_logsparsity": res_with,
        "without_logsparsity": res_without,
        "delta_auc_tune": delta_tune,
        "delta_auc_holdout": delta_holdout,
        "delta_auc_full": delta_full,
        "verdict": verdict,
    }

    save_summary_json_csv(summary, OUT_DIR)

    print("\n========== SINGLE-PAIR ABLATION SUMMARY ==========")
    print(
        f"with_logsparsity    | AUC_tune={res_with['auc_tune']:.6f} | "
        f"AUC_holdout={res_with['auc_holdout']:.6f} | "
        f"AUC_full={res_with['auc_full']:.6f} | "
        f"time={res_with['time_sec']:.2f}s"
    )
    print(
        f"without_logsparsity | AUC_tune={res_without['auc_tune']:.6f} | "
        f"AUC_holdout={res_without['auc_holdout']:.6f} | "
        f"AUC_full={res_without['auc_full']:.6f} | "
        f"time={res_without['time_sec']:.2f}s"
    )
    print(
        f"delta               | tune={delta_tune:+.6f} | "
        f"holdout={delta_holdout:+.6f} | "
        f"full={delta_full:+.6f}"
    )
    print(f"verdict             | {verdict}")
    print(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
