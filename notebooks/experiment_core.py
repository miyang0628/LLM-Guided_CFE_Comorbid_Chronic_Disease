"""
experiment_core.py
==================
Engine for the redesigned four-condition comparison.

Conditions
----------
  C0  pure         : unconstrained DiCE, Class-0 target, no fallback.
  C1  posthoc      : unconstrained DiCE, then DROP candidates that violate the
                     safety guardrail (post-hoc filtering). Feasibility = did any
                     candidate survive filtering.
  C2  pre_rule     : rule-based guardrail injected BEFORE search + stepwise
                     fallback (Class 0 -> 1 -> 2).
  C3  pre_llm      : LLM-derived guardrail injected before search + fallback.
                     (Cohort runs may pass a precomputed guardrail per patient.)

Primary metrics (claim b): feasibility rate, surviving valid candidates,
recourse diversity. Violation rate is reported only as a sanity check (0 by
construction for C1/C2/C3).
"""

import numpy as np
import pandas as pd
import dice_ml
import guardrail_core as gc


def _violation_flags(orig, row):
    """Return (anthro, energy, conflict) booleans for one candidate row."""
    o = {k: float(orig[k]) for k in
         ["BMI", "WaistCirc", "Weight", "Energy_kcal", "Sodium_mg", "Carb_g", "Sugar_g"]}
    anthro = not (row["BMI"] - o["BMI"] <= 0.01 and
                  row["WaistCirc"] - o["WaistCirc"] <= 0.01 and
                  row["Weight"] - o["Weight"] <= 0.01)
    energy = row["Energy_kcal"] < o["Energy_kcal"] * 0.70 - 1
    conflict = (row["Sodium_mg"] < o["Sodium_mg"] - 1 and
                (row["Carb_g"] > o["Carb_g"] + 1 or row["Sugar_g"] > o["Sugar_g"] + 1))
    return anthro, energy, conflict


def _candidate_violates_safety(orig, row, cur, X_FEATURES):
    """
    A candidate violates the SAFETY guardrail if it falls outside the Layer-1
    safe band. Used by C1 (post-hoc filtering) to decide which candidates to
    drop. Mirrors the pre-injection guardrail so C1 and C2 are comparable.
    """
    g = gc.build_rule_guardrails(cur, X_FEATURES, "aggressive")
    for f in X_FEATURES:
        lo, hi = g[f]
        v = float(row[f])
        if v < lo - 1e-6 or v > hi + 1e-6:
            return True
    return False


def _diversity(cf_df, df_stable, X_FEATURES):
    """Mean pairwise normalised L1 distance among surviving candidates."""
    if len(cf_df) < 2:
        return 0.0
    stds = df_stable[X_FEATURES].std().replace(0, 1).values
    M = cf_df[X_FEATURES].values / stds
    n = len(M)
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += np.abs(M[i] - M[j]).sum()
            cnt += 1
    return tot / cnt if cnt else 0.0


def _quality(cf_df, orig, df_stable, X_FEATURES):
    diffs = cf_df[X_FEATURES].values - orig[X_FEATURES].values
    n_changed = (np.abs(diffs) > 0.01).sum(axis=1).mean()
    stds = df_stable[X_FEATURES].std().replace(0, 1).values
    l1 = (np.abs(diffs) / stds).sum(axis=1).mean()
    va = ea = ca = 0
    for _, row in cf_df.iterrows():
        a, e, c = _violation_flags(orig, row)
        va += a; ea += e; ca += c
    n = len(cf_df)
    return {
        "n_changed_vars": round(float(n_changed), 2),
        "mean_l1_dist":   round(float(l1), 4),
        "diversity":      round(_diversity(cf_df, df_stable, X_FEATURES), 4),
        "anthro_viol":    va / n, "energy_viol": ea / n, "conflict_viol": ca / n,
        "any_viol":       int(va + ea + ca > 0),
    }


def _empty_result():
    return {"feasible": False, "achieved_class": None, "fallback_depth": None,
            "n_valid_cands": 0, "n_changed_vars": np.nan, "mean_l1_dist": np.nan,
            "diversity": np.nan, "anthro_viol": np.nan, "energy_viol": np.nan,
            "conflict_viol": np.nan, "any_viol": np.nan}


def make_dice(model, df_stable, X_FEATURES, TARGET_COL):
    d = dice_ml.Data(dataframe=df_stable[X_FEATURES + [TARGET_COL]],
                     continuous_features=X_FEATURES, outcome_name=TARGET_COL)
    m = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(d, m, method="genetic")


def eval_C0_pure(exp, query, df_stable, X_FEATURES, VARY, tries=5):
    """Unconstrained DiCE, Class 0 only."""
    res = _empty_result()
    orig = query.iloc[0]
    for _ in range(tries):
        try:
            cf = exp.generate_counterfactuals(
                query, total_CFs=4, desired_class=0,
                features_to_vary=VARY, proximity_weight=0.2, sparsity_weight=0.1)
            df = cf.cf_examples_list[0].final_cfs_df
            if df is not None and len(df) > 0:
                q = _quality(df.copy(), orig, df_stable, X_FEATURES)
                res.update({"feasible": True, "achieved_class": 0, "fallback_depth": 0,
                            "n_valid_cands": len(df), **q})
                return res
        except Exception:
            continue
    return res


def eval_C1_posthoc(exp, query, df_stable, X_FEATURES, VARY, cur, tries=5):
    """Unconstrained DiCE, then drop safety-violating candidates."""
    res = _empty_result()
    orig = query.iloc[0]
    for _ in range(tries):
        try:
            cf = exp.generate_counterfactuals(
                query, total_CFs=4, desired_class=0,
                features_to_vary=VARY, proximity_weight=0.2, sparsity_weight=0.1)
            df = cf.cf_examples_list[0].final_cfs_df
            if df is not None and len(df) > 0:
                keep = [not _candidate_violates_safety(orig, row, cur, X_FEATURES)
                        for _, row in df.iterrows()]
                surv = df[pd.Series(keep, index=df.index)].copy()
                if len(surv) > 0:
                    q = _quality(surv, orig, df_stable, X_FEATURES)
                    res.update({"feasible": True, "achieved_class": 0, "fallback_depth": 0,
                                "n_valid_cands": len(surv), **q})
                else:
                    # all candidates filtered out -> infeasible after filtering
                    res.update({"feasible": False, "n_valid_cands": 0})
                return res
        except Exception:
            continue
    return res


def eval_pre_injection(exp, query, df_stable, X_FEATURES, VARY, guardrails,
                       tries=5):
    """
    Guardrail injected before search + stepwise fallback (Class 0->1->2).
    Works for both C2 (rule guardrail) and C3 (LLM guardrail); the caller
    supplies the guardrail dict.
    """
    res = _empty_result()
    orig = query.iloc[0]
    for target_class in [0, 1, 2]:
        for _ in range(tries):
            try:
                cf = exp.generate_counterfactuals(
                    query, total_CFs=4, desired_class=target_class,
                    features_to_vary=VARY, permitted_range=guardrails,
                    proximity_weight=0.2, sparsity_weight=0.1)
                df = cf.cf_examples_list[0].final_cfs_df
                if df is not None and len(df) > 0:
                    q = _quality(df.copy(), orig, df_stable, X_FEATURES)
                    res.update({"feasible": True, "achieved_class": target_class,
                                "fallback_depth": target_class,
                                "n_valid_cands": len(df), **q})
                    return res
            except Exception:
                continue
    return res
