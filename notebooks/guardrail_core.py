"""
guardrail_core.py
=================
Shared guardrail logic for the redesigned experiments.

Three-layer variable structure
-------------------------------
  Layer 1 (SAFETY)   : absolute physiological floors + anthropometric
                       direction-coupling. ALWAYS enforced by code; neither the
                       LLM nor the rule baseline may relax these.
  Layer 2 (DISCRETION): "how far to move" ranges. Either filled by a fixed-ratio
                       RULE, or delegated to the LLM. This is the layer where the
                       LLM-vs-rule contrast (claim a) lives.
  Layer 3 (FIXED)    : non-modifiable variables, held at the patient's value.

Delegation level
----------------
  Controls how many Layer-2 variables are opened to discretion vs. pinned by
  Layer-1-style safety logic. Used for the delegation ablation.
    'conservative' : sodium and energy are pinned tightly by code; only
                     BMI / anthropometrics / fat / sugar are discretionary.
    'aggressive'   : only absolute safety floors are pinned; sodium-reduction
                     depth, energy target, and all diet composition are
                     discretionary.

Constraint-injection mode (for claim b)
---------------------------------------
  Not handled here; see the experiment notebooks. This module only *builds*
  a permitted_range; how it is used (pre-injection vs post-hoc filter) is the
  notebook's concern.
"""

import numpy as np


# Absolute physiological floors (Layer 1). These are the "cross a line and it is
# dangerous" values and are never delegated. Basis noted inline.
SAFETY = {
    "Energy_kcal_floor_abs": 500.0,    # never below 500 kcal
    "Sodium_mg_floor_abs":   800.0,    # WHO minimum safe sodium intake
    "Protein_g_floor_abs":   30.0,     # sarcopenia-prevention minimum
    "Potassium_mg_floor_abs": 500.0,   # cardiac-safety minimum
    "Carb_g_floor_abs":      30.0,     # brain-glucose minimum
    "Fiber_g_floor_abs":     5.0,      # gut-health minimum
    "anthro_max_reduction":  0.15,     # BMI/waist/weight: at most 15% reduction/step
}


def _clip_lo_hi(lo, hi, cur):
    """Guarantee lo <= hi; fall back to a tight consistent band on inversion."""
    if lo > hi:
        return round(cur * 0.85, 4), round(cur, 4)
    return round(lo, 4), round(hi, 4)


def apply_layer1_safety(guardrails, cur):
    """
    Enforce Layer-1 safety on an existing guardrail dict, in place.
    `cur` is a dict of the patient's current values.
    These override whatever Layer 2 proposed, wherever Layer 2 is less strict.
    """
    # Anthropometric direction coupling: BMI, WaistCirc, Weight must all be
    # reductions bounded by anthro_max_reduction, moving in the same direction.
    r = 1.0 - SAFETY["anthro_max_reduction"]
    for feat in ["BMI", "WaistCirc", "Weight"]:
        c = cur[feat]
        lo = max(guardrails.get(feat, [c * r, c])[0], c * r)
        hi = min(guardrails.get(feat, [c * r, c])[1], c)     # no increase
        guardrails[feat] = list(_clip_lo_hi(lo, hi, c))
    # Tie waist/BMI floor ratio to the weight floor ratio (constant height).
    wratio = guardrails["Weight"][0] / cur["Weight"]
    for feat in ["WaistCirc", "BMI"]:
        guardrails[feat][0] = max(guardrails[feat][0], round(cur[feat] * wratio, 4))
        guardrails[feat] = list(_clip_lo_hi(*guardrails[feat], cur[feat]))

    # Absolute nutritional floors (never delegated).
    guardrails["Energy_kcal"][0]  = max(guardrails["Energy_kcal"][0],
                                        SAFETY["Energy_kcal_floor_abs"])
    guardrails["Sodium_mg"][0]    = max(guardrails["Sodium_mg"][0],
                                        SAFETY["Sodium_mg_floor_abs"])
    guardrails["Protein_g"][0]    = max(guardrails["Protein_g"][0],
                                        SAFETY["Protein_g_floor_abs"])
    guardrails["Potassium_mg"][0] = max(guardrails["Potassium_mg"][0],
                                        SAFETY["Potassium_mg_floor_abs"])
    guardrails["Carb_g"][0]       = max(guardrails["Carb_g"][0],
                                        SAFETY["Carb_g_floor_abs"])
    guardrails["Fiber_g"][0]      = max(guardrails["Fiber_g"][0],
                                        SAFETY["Fiber_g_floor_abs"])

    # Energy / sodium ceilings are never above current (no increase permitted).
    guardrails["Energy_kcal"][1] = min(guardrails["Energy_kcal"][1], cur["Energy_kcal"])
    guardrails["Sodium_mg"][1]   = min(guardrails["Sodium_mg"][1], cur["Sodium_mg"])

    # Conflict floor for glycaemic safety: carb & sugar ceilings never exceed current.
    guardrails["Carb_g"][1]  = min(guardrails["Carb_g"][1],  cur["Carb_g"])
    guardrails["Sugar_g"][1] = min(guardrails["Sugar_g"][1], cur["Sugar_g"])

    for feat in list(guardrails):
        guardrails[feat] = list(_clip_lo_hi(*guardrails[feat], cur[feat]))
    return guardrails


def build_rule_guardrails(cur, X_FEATURES, delegation="aggressive"):
    """
    RULE baseline: fill Layer 2 with fixed-ratio bands, then enforce Layer 1.
    The delegation level controls how tightly sodium/energy are pinned, so that
    the rule baseline is directly comparable to the LLM at the same delegation.

    Returns permitted_range dict {feat: [lo, hi]}.
    """
    g = {}
    # Default wide band for every feature (Layer-2 fixed ratio).
    for feat in X_FEATURES:
        c = cur[feat]
        g[feat] = [max(c * 0.70, 0.0), c * 1.30]

    # Anthropometrics: fixed 15% reduction band.
    for feat in ["BMI", "WaistCirc", "Weight"]:
        c = cur[feat]
        g[feat] = [c * 0.85, c * 1.00]

    # Energy / sodium bands depend on delegation.
    if delegation == "conservative":
        # Pinned tightly by rule: fixed 70% energy floor, fixed 10% sodium cut.
        g["Energy_kcal"] = [max(cur["Energy_kcal"] * 0.70, 500.0), cur["Energy_kcal"]]
        g["Sodium_mg"]   = [max(cur["Sodium_mg"] * 0.60, 800.0),   cur["Sodium_mg"] * 0.90]
    else:  # aggressive
        # Wider rule band (still safe): energy down to floor, sodium down to floor.
        g["Energy_kcal"] = [max(cur["Energy_kcal"] * 0.70, 500.0), cur["Energy_kcal"]]
        g["Sodium_mg"]   = [max(cur["Sodium_mg"] * 0.50, 800.0),   cur["Sodium_mg"]]

    # Nutritional minimums (fixed ratios; Layer 1 tightens if needed).
    g["Protein_g"][0]    = max(cur["Protein_g"] * 0.60, 30.0)
    g["Potassium_mg"][0] = max(cur["Potassium_mg"] * 0.50, 500.0)
    g["Carb_g"]          = [max(cur["Carb_g"] * 0.20, 30.0), cur["Carb_g"]]
    g["Sugar_g"]         = [0.0, cur["Sugar_g"]]
    g["Fiber_g"][0]      = max(cur["Fiber_g"] * 0.30, 5.0)

    return apply_layer1_safety(g, cur)


def sanitize_llm_guardrails(llm_ranges, cur, X_FEATURES, delegation="aggressive"):
    """
    Take raw LLM-proposed ranges and enforce ONLY Layer 1 on top. Layer-2
    discretion from the LLM is preserved (this is what makes the LLM differ
    from the rule baseline). Missing features fall back to a wide safe band.

    Under 'conservative' delegation, sodium and energy are additionally pinned
    to the rule band regardless of the LLM (they are moved out of Layer 2).
    """
    g = {}
    for feat in X_FEATURES:
        c = cur[feat]
        if feat in llm_ranges:
            lo = max(float(llm_ranges[feat][0]), 0.0)
            hi = max(float(llm_ranges[feat][1]), lo)
        else:
            lo, hi = max(c * 0.70, 0.0), c * 1.30
        g[feat] = [lo, hi]

    if delegation == "conservative":
        # Remove sodium/energy from LLM discretion; pin to rule band.
        g["Energy_kcal"] = [max(cur["Energy_kcal"] * 0.70, 500.0), cur["Energy_kcal"]]
        g["Sodium_mg"]   = [max(cur["Sodium_mg"] * 0.60, 800.0),   cur["Sodium_mg"] * 0.90]

    return apply_layer1_safety(g, cur)


# ---------------------------------------------------------------------------
# Interval-difference metrics for claim (a): how far LLM ranges depart from rule
# ---------------------------------------------------------------------------
def interval_iou(a, b):
    """Intersection-over-union of two 1-D intervals [lo, hi]."""
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 1e-9 else 1.0


def guardrail_divergence(g_llm, g_rule, cur, features):
    """
    Per-feature divergence between an LLM guardrail and the rule guardrail for
    the same patient. Returns a dict of metrics per feature plus aggregates.
      width_ratio : LLM band width / rule band width
      lo_shift    : (LLM lo - rule lo) normalised by current value
      hi_shift    : (LLM hi - rule hi) normalised by current value
      iou         : interval overlap (1 = identical, 0 = disjoint)
    """
    out = {}
    ious = []
    for f in features:
        c = max(abs(cur[f]), 1e-6)
        wl = g_llm[f][1] - g_llm[f][0]
        wr = g_rule[f][1] - g_rule[f][0]
        iou = interval_iou(g_llm[f], g_rule[f])
        ious.append(iou)
        out[f] = {
            "width_ratio": round(wl / wr, 4) if wr > 1e-9 else np.nan,
            "lo_shift":    round((g_llm[f][0] - g_rule[f][0]) / c, 4),
            "hi_shift":    round((g_llm[f][1] - g_rule[f][1]) / c, 4),
            "iou":         round(iou, 4),
        }
    out["_mean_iou"] = round(float(np.mean(ious)), 4)
    return out
