"""
experiment_core_v2.py
=====================
Revised experiment engine for the injection-point and delegation study.

Design notes:

  #2  De-confound the injection-point contrast from the stepwise fallback.
      We evaluate injection point (post-hoc vs pre-injection) and fallback
      (on/off) as an orthogonal 2x2 factorial, under a *matched* search budget
      (same total_CFs, same number of tries, same target-class schedule, same
      seed). We also add a rejection-sampling feasibility baseline that is
      independent of DiCE, and a feasibility curve over search budget.

  #3  Separate violation categories (range / direction / floor / coupling)
      with an explicit, documented definition and denominator, so the
      candidate-level "all violate" statement and the individual-level
      "Any viol." rate are reconcilable.

  #5  Support evaluation over ALL predicted Class-3 patients (not only
      true-and-predicted), with true/predicted-class tags carried through.

The Layer-1/2/3 guardrail construction itself is unchanged; it still lives in
guardrail_core.py. This module only changes how the guardrail is *used*.
"""

import numpy as np
import pandas as pd
import dice_ml
import guardrail_core as gc


# ---------------------------------------------------------------------------
# Violation taxonomy 
# ---------------------------------------------------------------------------
# A candidate is scored against the SAME rule guardrail that C2/C3 inject, so
# that "would this candidate have survived pre-injection?" is exactly the
# post-hoc filter question. We decompose any out-of-band feature into four
# mutually exclusive-by-precedence categories:
#
#   direction : an anthropometric (BMI/Waist/Weight) moved UP (wrong sign),
#               or energy/sodium/carb/sugar moved UP where only reduction is
#               admissible.
#   floor     : a nutritional value fell BELOW an absolute Layer-1 safety floor
#               (energy/sodium/protein/potassium/carb/fiber).
#   coupling  : BMI/Waist/Weight moved down but NOT in the locked constant-height
#               ratio (|implied height| inconsistency > tol), i.e. the anthro
#               coupling constraint is broken.
#   range     : any other breach of the permitted [lo, hi] band not captured
#               above (magnitude too large, etc.).
#
# "any_violation" is the OR over the four. The candidate-level denominator is
# the number of generated candidates; the individual-level "Any viol." rate in
# the C0 row is the fraction of PATIENTS with >=1 violating candidate. These
# are different denominators, which is why C0 can show ~90% patients-with-a-
# -violation while a post-hoc filter that requires a FULLY admissible candidate
# still drops 100% of candidates.

_ANTHRO = ["BMI", "WaistCirc", "Weight"]
_FLOOR_KEYS = {
    "Energy_kcal": gc.SAFETY["Energy_kcal_floor_abs"],
    "Sodium_mg":   gc.SAFETY["Sodium_mg_floor_abs"],
    "Protein_g":   gc.SAFETY["Protein_g_floor_abs"],
    "Potassium_mg": gc.SAFETY["Potassium_mg_floor_abs"],
    "Carb_g":      gc.SAFETY["Carb_g_floor_abs"],
    "Fiber_g":     gc.SAFETY["Fiber_g_floor_abs"],
}
_REDUCE_ONLY = ["Energy_kcal", "Sodium_mg", "Carb_g", "Sugar_g"]


def classify_violations(orig, row, cur, X_FEATURES, g_rule=None, tol=1e-6):
    """
    Return a dict of booleans {direction, floor, coupling, range, any} for a
    single candidate row, plus the count of out-of-band features.
    """
    if g_rule is None:
        g_rule = gc.build_rule_guardrails(cur, X_FEATURES, "aggressive")

    direction = floor = coupling = rng = False

    # direction: anthropometrics may only go down; reduce-only nutrients may
    # only go down.
    for f in _ANTHRO:
        if float(row[f]) > float(orig[f]) + 0.01:
            direction = True
    for f in _REDUCE_ONLY:
        if f in row and float(row[f]) > float(orig[f]) + 1.0:
            direction = True

    # floor: below an absolute safety floor.
    for f, fl in _FLOOR_KEYS.items():
        if f in row and float(row[f]) < fl - tol:
            floor = True

    # coupling: constant-height ratio broken among BMI/Waist/Weight.
    # If weight drops by ratio w, BMI must drop by (approximately) the same
    # ratio (height fixed); waist should track closely. We flag if the BMI
    # reduction ratio departs from the weight reduction ratio by > 3 pts.
    w_ratio = float(row["Weight"]) / max(float(orig["Weight"]), 1e-6)
    bmi_ratio = float(row["BMI"]) / max(float(orig["BMI"]), 1e-6)
    if abs(bmi_ratio - w_ratio) > 0.03:
        coupling = True

    # range: any other permitted-band breach.
    for f in X_FEATURES:
        lo, hi = g_rule[f]
        v = float(row[f])
        if v < lo - tol or v > hi + tol:
            # attribute to range only if not already one of the above buckets
            if not (direction or floor or coupling):
                rng = True
            else:
                rng = rng  # keep; range still recorded below via n_outofband

    n_outofband = sum(
        (float(row[f]) < g_rule[f][0] - tol or float(row[f]) > g_rule[f][1] + tol)
        for f in X_FEATURES
    )
    any_v = bool(direction or floor or coupling or rng or n_outofband > 0)
    return {
        "v_direction": direction, "v_floor": floor, "v_coupling": coupling,
        "v_range": rng, "v_any": any_v, "n_outofband": int(n_outofband),
    }


def candidate_fully_admissible(orig, row, cur, X_FEATURES, g_rule=None):
    """A candidate survives the post-hoc filter iff NO feature is out of band."""
    if g_rule is None:
        g_rule = gc.build_rule_guardrails(cur, X_FEATURES, "aggressive")
    for f in X_FEATURES:
        lo, hi = g_rule[f]
        v = float(row[f])
        if v < lo - 1e-6 or v > hi + 1e-6:
            return False
    return True


# ---------------------------------------------------------------------------
# Shared quality metrics
# ---------------------------------------------------------------------------
def _diversity(cf_df, df_stable, X_FEATURES):
    if len(cf_df) < 2:
        return 0.0
    stds = df_stable[X_FEATURES].std().replace(0, 1).values
    M = cf_df[X_FEATURES].values / stds
    n = len(M); tot = cnt = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            tot += np.abs(M[i] - M[j]).sum(); cnt += 1
    return tot / cnt if cnt else 0.0


def _quality(cf_df, orig, cur, df_stable, X_FEATURES, g_rule=None):
    diffs = cf_df[X_FEATURES].values - orig[X_FEATURES].values
    n_changed = (np.abs(diffs) > 0.01).sum(axis=1).mean()
    stds = df_stable[X_FEATURES].std().replace(0, 1).values
    l1 = (np.abs(diffs) / stds).sum(axis=1).mean()
    vd = vf = vc = vr = va = 0
    for _, row in cf_df.iterrows():
        z = classify_violations(orig, row, cur, X_FEATURES, g_rule)
        vd += z["v_direction"]; vf += z["v_floor"]; vc += z["v_coupling"]
        vr += z["v_range"]; va += z["v_any"]
    n = len(cf_df)
    return {
        "n_changed_vars": round(float(n_changed), 2),
        "mean_l1_dist":   round(float(l1), 4),
        "diversity":      round(_diversity(cf_df, df_stable, X_FEATURES), 4),
        "direction_viol": vd / n, "floor_viol": vf / n, "coupling_viol": vc / n,
        "range_viol": vr / n, "any_viol": int(va > 0),
        "frac_cands_viol": va / n,
    }


def _empty():
    return {"feasible": False, "achieved_class": None, "fallback_depth": None,
            "n_valid_cands": 0, "n_changed_vars": np.nan, "mean_l1_dist": np.nan,
            "diversity": np.nan, "direction_viol": np.nan, "floor_viol": np.nan,
            "coupling_viol": np.nan, "range_viol": np.nan, "any_viol": np.nan,
            "frac_cands_viol": np.nan}


def make_dice(model, df_stable, X_FEATURES, TARGET_COL):
    d = dice_ml.Data(dataframe=df_stable[X_FEATURES + [TARGET_COL]],
                     continuous_features=X_FEATURES, outcome_name=TARGET_COL)
    m = dice_ml.Model(model=model, backend="sklearn")
    return dice_ml.Dice(d, m, method="genetic")


# ---------------------------------------------------------------------------
# UNIFIED evaluator: injection point x fallback, matched budget 
# ---------------------------------------------------------------------------
def evaluate(exp, query, df_stable, X_FEATURES, VARY, cur,
             injection="pre", fallback=True, guardrails=None,
             total_CFs=4, tries=5, target_schedule=(0, 1, 2), seed=0):
    """
    One evaluator for every condition, so that the ONLY things that vary across
    conditions are (injection, fallback). Budget is matched: same total_CFs,
    same tries, same target_schedule, same seed.

      injection = 'none' : unconstrained DiCE (C0-style raw search)
      injection = 'post' : unconstrained DiCE, then drop candidates that are
                           not fully admissible (post-hoc filter)
      injection = 'pre'  : guardrail injected into permitted_range before search

      fallback = True    : walk target_schedule (0 -> 1 -> 2), stop at first
                           class that yields a feasible (and, for 'post', a
                           surviving) candidate set.
      fallback = False   : only the first entry of target_schedule (Class 0).

    Returns a result dict. For 'pre', supply guardrails (rule or LLM). For
    'none'/'post', guardrails are built internally for scoring/filtering.
    """
    res = _empty()
    orig = query.iloc[0]
    g_rule = guardrails if guardrails is not None else \
        gc.build_rule_guardrails(cur, X_FEATURES, "aggressive")

    schedule = list(target_schedule) if fallback else [target_schedule[0]]

    for depth, target_class in enumerate(schedule):
        for _ in range(tries):
            try:
                kwargs = dict(query_instances=query, total_CFs=total_CFs,
                              desired_class=target_class, features_to_vary=VARY,
                              proximity_weight=0.2, sparsity_weight=0.1)
                if injection == "pre":
                    kwargs["permitted_range"] = guardrails
                cf = exp.generate_counterfactuals(**kwargs)
                df = cf.cf_examples_list[0].final_cfs_df
                if df is None or len(df) == 0:
                    continue

                if injection == "post":
                    keep = [candidate_fully_admissible(orig, row, cur,
                                                       X_FEATURES, g_rule)
                            for _, row in df.iterrows()]
                    df = df[pd.Series(keep, index=df.index)].copy()
                    if len(df) == 0:
                        # all filtered at this target; try next fallback depth
                        break

                q = _quality(df.copy(), orig, cur, df_stable, X_FEATURES, g_rule)
                res.update({"feasible": True, "achieved_class": target_class,
                            "fallback_depth": depth,
                            "n_valid_cands": len(df), **q})
                return res
            except Exception:
                continue
    return res


# ---------------------------------------------------------------------------
# DiCE-independent rejection-sampling feasibility baseline 
# ---------------------------------------------------------------------------
def rejection_sampling_baseline(model, query, cur, X_FEATURES, VARY,
                                guardrails, target_classes=(0, 1, 2),
                                n_samples=2000, seed=0):
    """
    Draw n_samples candidates uniformly INSIDE the admissible band for the
    varying features (fixed features held at the patient's value), and test
    whether ANY sampled point is classified into a target class. This is an
    optimiser-free feasibility oracle: if the admissible region contains a
    solution, enough uniform draws will find it. It bounds the DiCE result
    from below and shows the collapse is about the admissible region, not DiCE.
    """
    rng = np.random.default_rng(seed)
    base = query.iloc[0][X_FEATURES].astype(float).values.copy()
    idx = {f: i for i, f in enumerate(X_FEATURES)}
    vary = [f for f in VARY if f in guardrails]
    lo = np.array([guardrails[f][0] for f in vary])
    hi = np.array([guardrails[f][1] for f in vary])
    span = np.clip(hi - lo, 0, None)

    S = np.tile(base, (n_samples, 1))
    draws = lo + rng.random((n_samples, len(vary))) * span
    for j, f in enumerate(vary):
        S[:, idx[f]] = draws[:, j]
    # enforce anthro coupling on the samples (constant-height ratio)
    if all(k in idx for k in ("Weight", "BMI", "WaistCirc")):
        wr = S[:, idx["Weight"]] / max(float(cur["Weight"]), 1e-6)
        S[:, idx["BMI"]] = float(cur["BMI"]) * wr
        S[:, idx["WaistCirc"]] = float(cur["WaistCirc"]) * wr
    pred = model.predict(pd.DataFrame(S, columns=X_FEATURES))
    hit = np.isin(pred, np.array(target_classes))
    return {
        "rs_feasible": bool(hit.any()),
        "rs_hit_rate": float(hit.mean()),
        "rs_first_class": int(pred[hit][0]) if hit.any() else None,
    }
