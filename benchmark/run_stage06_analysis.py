from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
RESULTS = WORKSPACE / "results"
FIGURES = WORKSPACE / "figures"
NOTES = WORKSPACE / "notes"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)


def load_sources() -> dict:
    return {
        "manifest": read_json(RESULTS / "experiment_manifest.json"),
        "stage05": read_json(RESULTS / "stage05_execution_status.json"),
        "preflight": read_json(RESULTS / "open_dataset_real_run_preflight.json"),
        "gradient": pd.read_csv(RESULTS / "recalibration_gradient_pilot_summary.csv"),
        "gap": pd.read_csv(RESULTS / "recalibration_gradient_pilot_h3_gap_closure.csv"),
        "delong": pd.read_csv(RESULTS / "recalibration_gradient_pilot_delong.csv"),
        "h4_summary": pd.read_csv(RESULTS / "h4_size_ablation_pilot_summary.csv"),
        "h4_friedman_ece": pd.read_csv(RESULTS / "h4_friedman_test_per_size.csv"),
        "h4_friedman_brier": pd.read_csv(RESULTS / "h4_friedman_brier_test_per_size.csv"),
        "h4_nemenyi_ece": pd.read_csv(RESULTS / "h4_nemenyi_pairwise_ece.csv"),
        "h4_nemenyi_brier": pd.read_csv(RESULTS / "h4_nemenyi_pairwise_brier.csv"),
        "expanded_summary": pd.read_csv(RESULTS / "expanded_real_pilot_summary.csv"),
        "s4d_stmem": pd.read_csv(RESULTS / "s4d_vs_stmem_real_pilot_comparison.csv"),
    }


def write_evidence_inventory(src: dict) -> pd.DataFrame:
    manifest = src["manifest"]
    stage05 = src["stage05"]
    preflight = src["preflight"]

    rows = [
        {
            "evidence_class": "standardized_bundle",
            "scope": "60 result artifacts indexed in experiment_manifest.json",
            "claim_use": "analysis source of record",
            "limitations": "Bundle mixes synthetic validation, real pilots, preflight failures, and analysis artifacts.",
        },
        {
            "evidence_class": "synthetic_v3_validation",
            "scope": f"{stage05['available_result_evidence']['synthetic_phase_a_cells']} Phase A cells, "
            f"{stage05['available_result_evidence']['synthetic_phase_b_cells']} Phase B cells",
            "claim_use": "validates metric, recalibration, statistical-test, serialization, and figure plumbing",
            "limitations": "No empirical claim about ECG foundation-model behavior should cite this as evidence.",
        },
        {
            "evidence_class": "expanded_real_pilot",
            "scope": "PTB-XL and CODE-15%, 1000 records per dataset, S4-lite and ST-MEM, guarded Platt only",
            "claim_use": "feasibility evidence; identifies metric movement and AUROC-degradation risks",
            "limitations": "Not publication-grade; one FM, two datasets, small test sets, reduced S4-lite baseline.",
        },
        {
            "evidence_class": "configured_s4d_real_pilot",
            "scope": "PTB-XL and CODE-15%, 1000 records per dataset, constrained S4D and ST-MEM comparison",
            "claim_use": "stronger pilot comparator than S4-lite; useful for provisional gap-closure calculations",
            "limitations": "S4D training was still constrained; not full budget or full benchmark.",
        },
        {
            "evidence_class": "recalibration_gradient_pilot",
            "scope": "4 model-dataset cells x raw plus 3 recalibrators on fixed real pilot logits",
            "claim_use": "compares monotone Platt/temperature, isotonic, and learned MLP under small calibration sets",
            "limitations": "Calibration sets are about 200 examples; no large-n regime and only ST-MEM FM.",
        },
        {
            "evidence_class": "h4_size_micro_ablation",
            "scope": "560 rows: 4 model-dataset cells x 7 calibration sizes x 4 methods x 5 reps",
            "claim_use": "supports small-n method-ranking statements and MLP-overfitting risk",
            "limitations": "Maximum full calibration size is 201-203, far below the H4 large-n threshold of >10000.",
        },
        {
            "evidence_class": "full_benchmark_preflight",
            "scope": f"ready_to_launch={preflight['ready_to_launch']}; blockers={len(preflight['blockers'])}",
            "claim_use": "explains why original BenchECG-scale claim set is not yet settled",
            "limitations": "; ".join(preflight["blockers"]),
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "stage06_evidence_inventory.csv", index=False)
    return df


def write_real_pilot_deltas(src: dict) -> pd.DataFrame:
    gradient = src["gradient"].copy()
    raw = gradient[gradient["recal_method"] == "none"].copy()
    recal = gradient[gradient["recal_method"] != "none"].copy()
    key = ["dataset_id", "dataset_name", "model_id", "model_name"]
    merged = recal.merge(raw[key + ["auroc", "ece_ew", "brier", "sce"]], on=key, suffixes=("_post", "_raw"))
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            {
                "dataset_id": r["dataset_id"],
                "dataset_name": r["dataset_name"],
                "model_id": r["model_id"],
                "model_name": r["model_name"],
                "recal_method": r["recal_method"],
                "auroc_raw": r["auroc_raw"],
                "auroc_post": r["auroc_post"],
                "delta_auroc": r["auroc_post"] - r["auroc_raw"],
                "ece_ew_raw": r["ece_ew_raw"],
                "ece_ew_post": r["ece_ew_post"],
                "delta_ece_ew": r["ece_ew_post"] - r["ece_ew_raw"],
                "relative_ece_reduction": (r["ece_ew_raw"] - r["ece_ew_post"]) / r["ece_ew_raw"]
                if r["ece_ew_raw"] != 0
                else np.nan,
                "brier_raw": r["brier_raw"],
                "brier_post": r["brier_post"],
                "delta_brier": r["brier_post"] - r["brier_raw"],
                "sce_raw": r["sce_raw"],
                "sce_post": r["sce_post"],
                "delta_sce": r["sce_post"] - r["sce_raw"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "stage06_real_pilot_recalibration_deltas.csv", index=False)
    return out


def write_h4_summary(src: dict) -> pd.DataFrame:
    h4 = src["h4_summary"].copy()
    h4["cal_size_label"] = h4["cal_size"].astype(str)
    h4["cal_order"] = h4["cal_size_label"].map({"10": 10, "20": 20, "50": 50, "75": 75, "100": 100, "150": 150, "full": 200})
    avg = (
        h4.groupby(["cal_size_label", "cal_order", "method"], as_index=False)
        .agg(
            mean_ece=("ece_ew_mean", "mean"),
            mean_brier=("brier_mean", "mean"),
            mean_auroc=("auroc_mean", "mean"),
            cells=("ece_ew_mean", "size"),
        )
        .sort_values(["cal_order", "method"])
    )
    wide_ece = avg.pivot(index=["cal_size_label", "cal_order"], columns="method", values="mean_ece").reset_index()
    wide_brier = avg.pivot(index=["cal_size_label", "cal_order"], columns="method", values="mean_brier").reset_index()
    methods = ["temperature_scaling", "monotone_platt", "isotonic_regression", "learned_mlp_head"]
    rows = []
    ece_stats = src["h4_friedman_ece"].copy()
    brier_stats = src["h4_friedman_brier"].copy()
    for _, ece_row in wide_ece.iterrows():
        size_label = str(ece_row["cal_size_label"])
        brier_row = wide_brier[wide_brier["cal_size_label"].astype(str) == size_label].iloc[0]
        ece_f = ece_stats[ece_stats["cal_size"].astype(str) == size_label].iloc[0]
        brier_f = brier_stats[brier_stats["cal_size"].astype(str) == size_label].iloc[0]
        ece_vals = {m: float(ece_row[m]) for m in methods}
        brier_vals = {m: float(brier_row[m]) for m in methods}
        rows.append(
            {
                "cal_size": size_label,
                "effective_numeric_size": int(ece_row["cal_order"]),
                "mean_ece_temperature": ece_vals["temperature_scaling"],
                "mean_ece_platt": ece_vals["monotone_platt"],
                "mean_ece_isotonic": ece_vals["isotonic_regression"],
                "mean_ece_mlp": ece_vals["learned_mlp_head"],
                "best_ece_method": min(ece_vals, key=ece_vals.get),
                "mean_brier_temperature": brier_vals["temperature_scaling"],
                "mean_brier_platt": brier_vals["monotone_platt"],
                "mean_brier_isotonic": brier_vals["isotonic_regression"],
                "mean_brier_mlp": brier_vals["learned_mlp_head"],
                "best_brier_method": min(brier_vals, key=brier_vals.get),
                "ece_friedman_p": float(ece_f["friedman_pvalue"]),
                "brier_friedman_p": float(brier_f["friedman_pvalue"]),
                "mlp_ece_rank": float(ece_f["mean_ece_rank_mlp"]),
                "mlp_brier_rank": float(brier_f["mean_brier_rank_mlp"]),
                "mlp_ece_worst_fraction": float(ece_f["mlp_worst_fraction"]),
                "mlp_brier_worst_fraction": float(brier_f["mlp_worst_fraction"]),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("effective_numeric_size").reset_index(drop=True)
    out.to_csv(RESULTS / "stage06_h4_method_summary.csv", index=False)
    return out


def write_nemenyi_key_findings(src: dict) -> pd.DataFrame:
    rows = []
    for metric, df in [("ece_ew", src["h4_nemenyi_ece"]), ("brier", src["h4_nemenyi_brier"])]:
        sig = df[(df["sig_CD_05"]) | (df["wilcoxon_sig_bonf"])].copy()
        for _, r in sig.iterrows():
            lower_rank_method = r["method_a"] if r["mean_rank_a"] < r["mean_rank_b"] else r["method_b"]
            higher_rank_method = r["method_b"] if lower_rank_method == r["method_a"] else r["method_a"]
            rows.append(
                {
                    "metric": metric,
                    "cal_size": r["cal_size"],
                    "better_method_by_rank": lower_rank_method,
                    "worse_method_by_rank": higher_rank_method,
                    "rank_diff": r["rank_diff"],
                    "sig_cd_05": bool(r["sig_CD_05"]),
                    "wilcoxon_p": r["wilcoxon_p"],
                    "wilcoxon_sig_bonf": bool(r["wilcoxon_sig_bonf"]),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_cal_order"] = out["cal_size"].astype(str).map(
            {"10": 10, "20": 20, "50": 50, "75": 75, "100": 100, "150": 150, "full": 200}
        )
        out = out.sort_values(["metric", "_cal_order", "better_method_by_rank", "worse_method_by_rank"]).drop(
            columns=["_cal_order"]
        )
    out.to_csv(RESULTS / "stage06_nemenyi_key_pairs.csv", index=False)
    return out


def write_claim_tables(src: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    gap = src["gap"]
    delong = src["delong"]
    h4_ece = src["h4_friedman_ece"]
    h4_brier = src["h4_friedman_brier"]

    best_gap = gap.loc[gap.groupby("dataset_id")["ece_gap_closure"].idxmax()][
        ["dataset_id", "dataset_name", "recal_method", "ece_gap_closure"]
    ]
    h6_nominal = delong[delong["nominal_significant_degradation"] == True]
    n_h6_nominal = len(h6_nominal)
    n_mlp_cd_ece = len(
        src["h4_nemenyi_ece"][
            (src["h4_nemenyi_ece"]["sig_CD_05"])
            & (
                (src["h4_nemenyi_ece"]["method_a"] == "learned_mlp_head")
                | (src["h4_nemenyi_ece"]["method_b"] == "learned_mlp_head")
            )
        ]
    )
    n_mlp_cd_brier = len(
        src["h4_nemenyi_brier"][
            (src["h4_nemenyi_brier"]["sig_CD_05"])
            & (
                (src["h4_nemenyi_brier"]["method_a"] == "learned_mlp_head")
                | (src["h4_nemenyi_brier"]["method_b"] == "learned_mlp_head")
            )
        ]
    )
    verdict_rows = [
        {
            "hypothesis": "H1",
            "current_verdict": "unresolved; not supported by current real evidence",
            "strongest_supporting_evidence": "Synthetic H1 pipeline output behaves as designed.",
            "contradictory_or_limiting_evidence": "Real pilot has only ST-MEM plus constrained S4D/S4-lite on two datasets; ST-MEM outperforms S4D on AUROC and Brier in both available real pilot datasets.",
            "writing_guidance": "Do not claim ECG FMs are worse calibrated than S4 until all five FMs and full S4D are run with bootstrap tests.",
        },
        {
            "hypothesis": "H2",
            "current_verdict": "unresolved; no real support for net underconfidence",
            "strongest_supporting_evidence": "Synthetic SCE direction analysis validates the sign-analysis plumbing.",
            "contradictory_or_limiting_evidence": "Raw ST-MEM SCE is positive on PTB-XL and CODE-15% in the real pilot, not negative under the approved sign convention.",
            "writing_guidance": "Keep underconfidence as a hypothesis, not a result; report real SCE signs neutrally.",
        },
        {
            "hypothesis": "H3",
            "current_verdict": "pilot-level support for ECE closure only; not publication-grade",
            "strongest_supporting_evidence": "Best ST-MEM ECE gap closure vs configured S4D exceeds 1.0 on PTB-XL and CODE-15%; best methods are "
            + "; ".join(
                f"{r.dataset_name}: {r.recal_method}={r.ece_gap_closure:.2f}" for r in best_gap.itertuples()
            )
            + ".",
            "contradictory_or_limiting_evidence": "MLP has negative ECE gap closure on CODE-15%; all Brier gap-closure rows are excluded because ST-MEM already beats S4D by Brier before recalibration.",
            "writing_guidance": "Frame as encouraging feasibility evidence for ECE, not proof of supervised-baseline parity recovery.",
        },
        {
            "hypothesis": "H4",
            "current_verdict": "small-n MLP underperformance supported; large-n MLP superiority untested and unsupported",
            "strongest_supporting_evidence": f"At n=20, Friedman is significant for ECE (p={h4_ece.loc[h4_ece['cal_size'].astype(str)=='20','friedman_pvalue'].iloc[0]:.4g}) and Brier (p={h4_brier.loc[h4_brier['cal_size'].astype(str)=='20','friedman_pvalue'].iloc[0]:.4g}); MLP is CD-significantly worse in {n_mlp_cd_ece} ECE and {n_mlp_cd_brier} Brier MLP-involving pair(s).",
            "contradictory_or_limiting_evidence": "The largest real calibration size is about 200, far below the approved large-n threshold (>10000); Platt is the most stable Brier winner.",
            "writing_guidance": "Split the claim: retain small-calibration MLP caution; remove or label the large-n MLP-superiority claim as untested.",
        },
        {
            "hypothesis": "H5",
            "current_verdict": "unresolved",
            "strongest_supporting_evidence": "Synthetic rank-correlation code works.",
            "contradictory_or_limiting_evidence": "Only two real datasets and one FM probe are available; no cross-model or four-dataset generalization test is possible.",
            "writing_guidance": "Avoid cross-dataset generalization claims; use pilot results only to motivate full execution.",
        },
        {
            "hypothesis": "H6",
            "current_verdict": "method-dependent; monotone recalibration looks safe in pilot, non-monotone methods remain risky",
            "strongest_supporting_evidence": "Monotone Platt/temperature preserves AUROC exactly in the gradient pilot DeLong rows.",
            "contradictory_or_limiting_evidence": f"Isotonic and MLP produce {n_h6_nominal} nominal real-pilot AUROC-degradation flags before multiplicity correction; synthetic H6 also has corrected degradation cases.",
            "writing_guidance": "State AUROC preservation only for monotone transforms in this pilot; require corrected DeLong tests before any general H6 claim.",
        },
    ]
    verdicts = pd.DataFrame(verdict_rows)
    verdicts.to_csv(RESULTS / "stage06_hypothesis_verdicts.csv", index=False)

    claim_rows = [
        {
            "claim_area": "C1: calibration blind spot",
            "support_status": "supported as literature motivation",
            "confidence": "high for novelty framing; independent of pilot performance",
            "evidence_basis": "Prior stages verified that major ECG FM benchmarks omit calibration metrics.",
            "avoid": "Do not imply current experiments already provide the full systematic audit.",
        },
        {
            "claim_area": "C2: lightweight recalibration closes the supervised-baseline gap",
            "support_status": "not yet supported for the paper claim",
            "confidence": "low-to-moderate pilot feasibility only",
            "evidence_basis": "ST-MEM ECE gap closure is strong in two pilot datasets, but Brier parity is undefined and MLP can fail.",
            "avoid": "Do not call recalibration deployable or baseline-parity restoring without full real benchmark and H6 tests.",
        },
        {
            "claim_area": "C3: S4 remains competitive",
            "support_status": "unresolved; not demonstrated by the current pilot",
            "confidence": "low",
            "evidence_basis": "Configured S4D is below ST-MEM on AUROC and Brier in both real pilot datasets, but training budget is constrained.",
            "avoid": "Do not claim S4 superiority or competitiveness from the current runs.",
        },
        {
            "claim_area": "Recalibration method ranking",
            "support_status": "pilot-supported for small calibration sets",
            "confidence": "moderate within pilot scope",
            "evidence_basis": "H4 micro-ablation gives repeated blocks and Nemenyi/Wilcoxon checks; Platt is consistently strong and MLP is poor at n=20.",
            "avoid": "Do not extrapolate to n>10000 or full production-scale calibration.",
        },
        {
            "claim_area": "AUROC preservation",
            "support_status": "supported only for monotone transforms in pilot",
            "confidence": "moderate for pilot monotone Platt/temperature; low for isotonic and MLP",
            "evidence_basis": "Monotone method is rank-preserving; real DeLong rows show no monotone flags, while non-monotone methods show nominal flags.",
            "avoid": "Do not state post-hoc recalibration is generally discrimination-preserving.",
        },
        {
            "claim_area": "Full BenchECG head-to-head benchmark",
            "support_status": "blocked",
            "confidence": "high that current bundle cannot answer it",
            "evidence_basis": "Preflight reports missing CPSC metadata and missing/unloadable checkpoints/packages for all five FMs in the full runner.",
            "avoid": "Do not present pilot outputs as the requested five-FM, four-dataset benchmark.",
        },
    ]
    claims = pd.DataFrame(claim_rows)
    claims.to_csv(RESULTS / "stage06_claim_support_matrix.csv", index=False)
    return verdicts, claims


def write_note(evidence: pd.DataFrame, verdicts: pd.DataFrame, h4_summary: pd.DataFrame, deltas: pd.DataFrame, key_pairs: pd.DataFrame) -> None:
    best_stmem = deltas[deltas["model_id"] == "M2_stmem_probe"].copy()
    best_stmem_ece = best_stmem.sort_values(["dataset_id", "ece_ew_post"]).groupby("dataset_id").head(1)
    h4_n20 = h4_summary[h4_summary["cal_size"].astype(str) == "20"].iloc[0]
    h4_full = h4_summary[h4_summary["cal_size"].astype(str) == "full"].iloc[0]
    lines = [
        "# Stage 06 Analysis Interpretation",
        "",
        "## Evidence Boundary",
        "The standardized bundle is ready for analysis, but it does not contain the full requested BenchECG benchmark. "
        "The current real ECG evidence covers PTB-XL and CODE-15% only, with a verified ST-MEM frozen probe and constrained S4D/S4-lite baselines. "
        "Synthetic v3 artifacts are useful for software validation only.",
        "",
        "## Main Interpretive Findings",
        "1. H1 remains unresolved. The real pilot does not show S4 superiority; ST-MEM exceeds configured S4D on AUROC and Brier on both real pilot datasets. "
        "Because the S4D run is constrained and the full five-FM benchmark is blocked, this is not a falsification of H1.",
        "2. H2 remains unresolved. ST-MEM signed calibration error is positive on the two real pilot datasets, so current real evidence does not support the pre-specified underconfidence direction.",
        "3. H3 has encouraging but narrow ECE evidence. Best ST-MEM ECE gap closure exceeds 1.0 in both real pilot datasets, but Brier gap closure is excluded because ST-MEM already beats S4D by Brier before recalibration.",
        "4. H4 should be narrowed. The small-n MLP-underperformance result is the strongest current empirical signal. At n=20, MLP has mean ECE rank "
        f"{h4_n20['mlp_ece_rank']:.2f} and mean Brier rank {h4_n20['mlp_brier_rank']:.2f}; lower rank is better. "
        "The large-n MLP-superiority half is untested because the largest full calibration size is about 200, not >10000.",
        "5. H6 is method-dependent. Monotone Platt/temperature preserves rankings by construction and has no real-pilot DeLong flags; isotonic and learned MLP produce nominal AUROC-degradation flags and require corrected DeLong scrutiny.",
        "",
        "## Strong Conclusions",
        "- The framework can compute and serialize the required discrimination, calibration, recalibration, gap-closure, H4, and DeLong outputs.",
        "- For small calibration sets in the real pilot, a learned MLP head is risky and often inferior to Platt/temperature on both ECE and Brier.",
        "- Monotonicity is a practical safety property for post-hoc recalibration when AUROC preservation is a hard requirement.",
        "",
        "## Weak or Unsupported Claims",
        "- The current bundle cannot support a claim about five ECG FMs across four datasets.",
        "- The current bundle cannot support a general claim that ECG FMs are underconfident.",
        "- The current bundle cannot support a general claim that recalibration recovers supervised-baseline parity.",
        "- The current bundle cannot support MLP superiority at large calibration sizes.",
        "",
        "## Writing Guidance",
        "The writing stage should foreground the verified calibration blind spot and the implemented evaluation framework. "
        "Results should be framed as pilot evidence plus a well-supported small-calibration warning, not as the final JMLR-scale benchmark. "
        "The safest empirical narrative is: monotone recalibration is stable and sometimes useful; non-parametric and learned recalibrators can improve ECE but can alter ranking and need DeLong checks.",
        "",
        "## Key Quantitative Anchors",
    ]
    for r in best_stmem_ece.itertuples():
        lines.append(
            f"- {r.dataset_name}: best ST-MEM pilot recalibrator by ECE was {r.recal_method}, "
            f"ECE {r.ece_ew_raw:.4f} to {r.ece_ew_post:.4f}, AUROC delta {r.delta_auroc:.4f}."
        )
    lines.append(
        f"- H4 n=20: best ECE method was {h4_n20['best_ece_method']}; best Brier method was {h4_n20['best_brier_method']}."
    )
    lines.append(
        f"- H4 full (~200): best ECE method was {h4_full['best_ece_method']}; best Brier method was {h4_full['best_brier_method']}."
    )
    if not key_pairs.empty:
        sig_counts = key_pairs.groupby("metric").size().to_dict()
        lines.append(f"- Nemenyi/Wilcoxon key significant pairs: {sig_counts}.")
    lines.append("")
    lines.append("## Evidence Inventory")
    for r in evidence.itertuples():
        lines.append(f"- {r.evidence_class}: {r.scope}. Use: {r.claim_use}. Limitation: {r.limitations}")
    (NOTES / "stage06_analysis_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_hypothesis_map(verdicts: pd.DataFrame) -> None:
    status_order = {
        "supported": 3,
        "pilot-level support for ECE closure only; not publication-grade": 2,
        "small-n MLP underperformance supported; large-n MLP superiority untested and unsupported": 2,
        "method-dependent; monotone recalibration looks safe in pilot, non-monotone methods remain risky": 1,
        "unresolved; not supported by current real evidence": 0,
        "unresolved; no real support for net underconfidence": 0,
        "unresolved": 0,
    }
    scores = [status_order.get(v, 0) for v in verdicts["current_verdict"]]
    colors = {0: "#b8b8b8", 1: "#e6a23c", 2: "#4c78a8", 3: "#2f855a"}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(verdicts))
    ax.barh(y, scores, color=[colors[s] for s in scores], height=0.62)
    ax.set_yticks(y, verdicts["hypothesis"])
    ax.set_xlim(0, 3.2)
    ax.set_xticks([0, 1, 2, 3], ["unresolved", "mixed", "pilot support", "supported"])
    ax.set_title("Stage 06 claim support by current evidence")
    ax.grid(axis="x", alpha=0.25)
    for idx, (score, verdict) in enumerate(zip(scores, verdicts["current_verdict"])):
        ax.text(min(score + 0.05, 3.05), idx, verdict.split(";")[0], va="center", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "stage06_hypothesis_support_map.png", dpi=180)
    fig.savefig(FIGURES / "stage06_hypothesis_support_map.pdf")
    plt.close(fig)


def plot_h4_rank_profile(src: dict) -> None:
    ece = src["h4_friedman_ece"].copy()
    brier = src["h4_friedman_brier"].copy()
    methods = [
        ("temperature", "mean_ece_rank_temperature", "mean_brier_rank_temperature", "#4c78a8"),
        ("platt", "mean_ece_rank_platt", "mean_brier_rank_platt", "#2f855a"),
        ("isotonic", "mean_ece_rank_isotonic", "mean_brier_rank_isotonic", "#8f63b8"),
        ("mlp", "mean_ece_rank_mlp", "mean_brier_rank_mlp", "#c2410c"),
    ]
    x = np.arange(len(ece))
    labels = ece["cal_size"].astype(str).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for label, ece_col, brier_col, color in methods:
        axes[0].plot(x, ece[ece_col], marker="o", label=label, color=color, linewidth=2)
        axes[1].plot(x, brier[brier_col], marker="o", label=label, color=color, linewidth=2)
    for ax, title in zip(axes, ["ECE method ranks", "Brier method ranks"]):
        ax.set_xticks(x, labels)
        ax.set_xlabel("Calibration size")
        ax.set_ylim(3.6, 1.2)
        ax.grid(alpha=0.25)
        ax.set_title(title)
    axes[0].set_ylabel("Mean rank (lower is better)")
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("H4 pilot: MLP is weak in small calibration sets; large-n is untested", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage06_h4_rank_profile.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "stage06_h4_rank_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_real_pilot_deltas(deltas: pd.DataFrame) -> None:
    subset = deltas.copy()
    subset["cell"] = subset["dataset_name"].str.replace("%", "pct", regex=False) + "\n" + subset["model_id"].str.replace("_", " ")
    methods = subset["recal_method"].unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False)
    for ax, metric, title in [
        (axes[0], "delta_ece_ew", "Delta ECE after recalibration"),
        (axes[1], "delta_auroc", "Delta AUROC after recalibration"),
    ]:
        pivot = subset.pivot_table(index="cell", columns="recal_method", values=metric, aggfunc="mean")
        x = np.arange(len(pivot.index))
        width = 0.24
        for i, m in enumerate(methods):
            if m in pivot:
                ax.bar(x + (i - 1) * width, pivot[m], width=width, label=m)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, pivot.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Post minus raw")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage06_real_pilot_recalibration_deltas.png", dpi=180)
    fig.savefig(FIGURES / "stage06_real_pilot_recalibration_deltas.pdf")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    src = load_sources()
    evidence = write_evidence_inventory(src)
    deltas = write_real_pilot_deltas(src)
    h4_summary = write_h4_summary(src)
    key_pairs = write_nemenyi_key_findings(src)
    verdicts, claims = write_claim_tables(src)
    write_note(evidence, verdicts, h4_summary, deltas, key_pairs)
    plot_hypothesis_map(verdicts)
    plot_h4_rank_profile(src)
    plot_real_pilot_deltas(deltas)
    manifest = {
        "generated_by": "workspace/code/run_stage06_analysis.py",
        "inputs": [
            "workspace/results/experiment_manifest.json",
            "workspace/results/stage05_execution_status.json",
            "workspace/results/recalibration_gradient_pilot_summary.csv",
            "workspace/results/recalibration_gradient_pilot_h3_gap_closure.csv",
            "workspace/results/recalibration_gradient_pilot_delong.csv",
            "workspace/results/h4_size_ablation_pilot_summary.csv",
            "workspace/results/h4_friedman_test_per_size.csv",
            "workspace/results/h4_friedman_brier_test_per_size.csv",
            "workspace/results/h4_nemenyi_pairwise_ece.csv",
            "workspace/results/h4_nemenyi_pairwise_brier.csv",
        ],
        "outputs": [
            "workspace/results/stage06_evidence_inventory.csv",
            "workspace/results/stage06_real_pilot_recalibration_deltas.csv",
            "workspace/results/stage06_h4_method_summary.csv",
            "workspace/results/stage06_nemenyi_key_pairs.csv",
            "workspace/results/stage06_hypothesis_verdicts.csv",
            "workspace/results/stage06_claim_support_matrix.csv",
            "workspace/notes/stage06_analysis_interpretation.md",
            "workspace/figures/stage06_hypothesis_support_map.png",
            "workspace/figures/stage06_hypothesis_support_map.pdf",
            "workspace/figures/stage06_h4_rank_profile.png",
            "workspace/figures/stage06_h4_rank_profile.pdf",
            "workspace/figures/stage06_real_pilot_recalibration_deltas.png",
            "workspace/figures/stage06_real_pilot_recalibration_deltas.pdf",
        ],
    }
    (RESULTS / "stage06_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
