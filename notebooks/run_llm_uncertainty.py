"""
run_llm_uncertainty.py
======================
Uncertainty reporting: replace the p-value-centric reporting of the LLM tier experiment
with patient-level bootstrap confidence intervals and effect sizes, computed
from the cached per-patient IoU / breach data (results/tables/llm_tier_divergence.csv).

This is a re-analysis of existing per-patient outputs; it makes no new API calls.
The repeated-call (same patient, multiple samples) variance decomposition that
a same-patient repeated-call protocol also requires fresh API access and is documented in the
manuscript as a disclosed protocol item, not fabricated here.

Outputs:
  results/tables/llm_tier_uncertainty.csv   (tidy CI table)
  results/tables/table_tier_divergence.tex  (rewritten: CIs replace p-values)
"""
import os
import numpy as np, pandas as pd

TAB = "../results/tables"
rng = np.random.default_rng(20260911)
B = 10000

df = pd.read_csv(f"{TAB}/llm_tier_divergence.csv")
TIERS = ["weak", "mid", "strong"]
MODEL = {t: df[df.tier == t]["model"].iloc[0] for t in TIERS}


def boot_ci(x, stat=np.mean, B=B, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, n, size=(B, n))
    bs = stat(x[idx], axis=1)
    return (float(stat(x)),
            float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


def paired_boot_diff_ci(x, y, B=B, alpha=0.05):
    """Bootstrap CI for mean(x) - mean(y) on paired samples (same patients)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    bs = d[idx].mean(axis=1)
    return (float(d.mean()),
            float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))


def cohen_dz(x, y):
    """Paired Cohen's dz effect size."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[~np.isnan(d)]
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan


def one_sample_dz(x, mu=1.0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float((x.mean() - mu) / x.std(ddof=1)) if x.std(ddof=1) > 0 else np.nan


# ---- 1. Per-tier IoU vs rule: mean, 95% CI, and effect size vs identity(1.0)
rows = []
piv = df.pivot_table(index="pid", columns="tier", values="mean_iou")
for t in TIERS:
    x = df[df.tier == t]["mean_iou"].values
    m, lo, hi = boot_ci(x)
    # distance from identity (rule reproduction would give IoU = 1)
    dz = one_sample_dz(x, mu=1.0)
    br = df[df.tier == t]["raw_safety_breaches"].values
    bm, blo, bhi = boot_ci(br)
    fe = df[df.tier == t]["feasible"].astype(float).values
    fm, flo, fhi = boot_ci(fe)
    rows.append({"tier": t, "model": MODEL[t],
                 "iou_mean": m, "iou_lo": lo, "iou_hi": hi,
                 "iou_dz_vs_identity": dz,
                 "breach_mean": bm, "breach_lo": blo, "breach_hi": bhi,
                 "feas_mean": fm, "feas_lo": flo, "feas_hi": fhi,
                 "n": int(np.sum(~np.isnan(x)))})
tier_tbl = pd.DataFrame(rows)

# ---- 2. Pairwise tier differences in IoU (paired bootstrap + dz)
pairs = [("weak", "mid"), ("weak", "strong"), ("mid", "strong")]
prows = []
for a, b in pairs:
    sub = piv[[a, b]].dropna()
    dmean, dlo, dhi = paired_boot_diff_ci(sub[a].values, sub[b].values)
    prows.append({"contrast": f"{a}-{b}", "diff_mean": dmean,
                  "diff_lo": dlo, "diff_hi": dhi,
                  "dz": cohen_dz(sub[a].values, sub[b].values),
                  "n_pairs": len(sub)})
pair_tbl = pd.DataFrame(prows)

tier_tbl.to_csv(f"{TAB}/llm_tier_uncertainty.csv", index=False)
pair_tbl.to_csv(f"{TAB}/llm_tier_pairwise.csv", index=False)

print("=== Tier-level IoU vs rule (mean [95% bootstrap CI]) ===")
for _, r in tier_tbl.iterrows():
    print(f"  {r.tier:6s} {r.model:14s} IoU {r.iou_mean:.3f} "
          f"[{r.iou_lo:.3f}, {r.iou_hi:.3f}]  dz_vs_identity {r.iou_dz_vs_identity:+.2f}  "
          f"breaches {r.breach_mean:.2f} [{r.breach_lo:.2f}, {r.breach_hi:.2f}]  "
          f"feas {r.feas_mean:.3f} [{r.feas_lo:.3f}, {r.feas_hi:.3f}]")
print("\n=== Pairwise IoU contrasts (paired bootstrap) ===")
for _, r in pair_tbl.iterrows():
    print(f"  {r.contrast:14s} Δ {r.diff_mean:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]  "
          f"dz {r.dz:+.2f}  (n={r.n_pairs})")

# ---- 3. Rewrite the tier-divergence table with CIs instead of p-values
def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

tex = r"""\begin{table}[ht]
\centering
\caption{LLM guardrail behaviour across model tiers on all 155 model-confirmed
Class~3 patients. Interval IoU is measured against the fixed-ratio rule baseline
on delegated (Layer-2) variables (lower $=$ more patient-specific); raw safety
breaches count how often the model's pre-correction range would have crossed an
absolute safety floor (higher $=$ more reliant on the code-level guardrail).
Brackets are 95\% patient-level bootstrap confidence intervals
($B=10{,}000$). Layer-1 floors are enforced identically for every tier.}
\label{tab:tier_divergence}
\begin{tabular}{lccc}
\toprule
Tier (model) & Mean IoU vs rule [95\% CI] & Raw safety breaches [95\% CI] & Feasibility [95\% CI] \\
\midrule
"""
disp = {"weak": "weak", "mid": "mid", "strong": "strong"}
for _, r in tier_tbl.iterrows():
    tex += (f"  {disp[r.tier]} ({r.model}) & "
            f"{fmt(r.iou_mean, r.iou_lo, r.iou_hi)} & "
            f"{fmt(r.breach_mean, r.breach_lo, r.breach_hi, 2)} & "
            f"{fmt(r.feas_mean*100, r.feas_lo*100, r.feas_hi*100, 1)}\\% \\\\\n")
tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
with open(f"{TAB}/table_tier_divergence.tex", "w") as f:
    f.write(tex)
print("\nWROTE table_tier_divergence.tex, llm_tier_uncertainty.csv, llm_tier_pairwise.csv")
