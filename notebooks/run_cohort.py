"""
run_cohort.py
===============
Runs the de-confounded factorial (injection point x fallback) with a matched
search budget, the rejection-sampling baseline, and the separated violation
taxonomy, over BOTH cohorts:
  cohort = 'confirmed'  : true Class 3 AND predicted Class 3 (original n=155)
  cohort = 'predicted'  : ALL predicted Class 3 

Writes long-format results to results/tables/cohort.csv.
"""
import os, time, warnings, joblib, sys
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# Silence DiCE's per-call tqdm bars (they flood the log).
os.environ["TQDM_DISABLE"] = "1"
try:
    import tqdm as _tqdm
    from functools import partialmethod
    _tqdm.tqdm.__init__ = partialmethod(_tqdm.tqdm.__init__, disable=True)
except Exception:
    pass

import guardrail_core as gc
import experiment_core_v2 as ec2

DATA = "../data"; TAB = "../results/tables"
os.makedirs(TAB, exist_ok=True)

df_final = joblib.load(f"{DATA}/df_final.pkl")
ac = joblib.load(f"{DATA}/agent_config.pkl")
X, T, VARY = ac["X_features"], ac["target_col"], ac["vary_features"]

CONDITIONS = [
    # (name, injection, fallback)
    ("C0_pure",        "none", False),
    ("C1_post_nofb",   "post", False),   # post-hoc, NO fallback  (original C1)
    ("C1b_post_fb",    "post", True),    # post-hoc, WITH fallback 
    ("C2a_pre_nofb",   "pre",  False),   # pre-injection, NO fallback 
    ("C2_pre_fb",      "pre",  True),    # pre-injection, WITH fallback (original C2)
]

KEEP = ["feasible", "achieved_class", "fallback_depth", "n_valid_cands",
        "n_changed_vars", "mean_l1_dist", "diversity", "direction_viol",
        "floor_viol", "coupling_viol", "range_viol", "any_viol",
        "frac_cands_viol"]


def run_gender(code, name, model_file, val_file, seed=0):
    model = joblib.load(f"{DATA}/{model_file}")
    val = joblib.load(f"{DATA}/{val_file}")
    df_stable = df_final[df_final["Sex"] == code].copy().astype(float)
    pred = model.predict(val["X_val"])
    exp = ec2.make_dice(model, df_stable, X, T)

    predicted_idx = val["X_val"][pred == 3].index.tolist()
    # resume: skip patients already fully done in the checkpoint
    ckpt = f"{TAB}/_ckpt_{name.lower()}.csv"
    rows = []
    done = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        cnt = prev.groupby("pid")["cond"].nunique()
        done = set(cnt[cnt >= len(CONDITIONS)].index.tolist())
        rows = prev[prev.pid.isin(done)].to_dict("records")
        print(f"{name}: resuming, {len(done)} patients already done", flush=True)
    true_arr = val["y_val"]
    t0 = time.time()
    todo = [p for p in predicted_idx if p not in done]
    for k, pid in enumerate(todo):
        q = val["X_val"].loc[[pid]][X]
        cur = {f: float(q.iloc[0][f]) for f in X}
        true_c = int(true_arr.loc[pid]) if pid in true_arr.index else int(true_arr[k])
        is_confirmed = (true_c == 3)
        g_rule = gc.build_rule_guardrails(cur, X, "aggressive")

        rs = ec2.rejection_sampling_baseline(model, q, cur, X, VARY, g_rule,
                                             n_samples=2000, seed=seed)

        for cname, inj, fb in CONDITIONS:
            gr = g_rule if inj == "pre" else None
            r = ec2.evaluate(exp, q, df_stable, X, VARY, cur,
                             injection=inj, fallback=fb, guardrails=gr,
                             total_CFs=4, tries=5, seed=seed)
            rows.append({
                "pid": pid, "gender": name, "true_class": true_c,
                "pred_class": 3, "confirmed": is_confirmed, "cond": cname,
                "rs_feasible": rs["rs_feasible"], "rs_hit_rate": rs["rs_hit_rate"],
                **{kk: r[kk] for kk in KEEP}})
        if (k + 1) % 5 == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            print(f"  {name} {len(done)+k+1}/{len(predicted_idx)} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    df_g = pd.DataFrame(rows)
    df_g.to_csv(ckpt, index=False)
    print(f"{name}: {len(predicted_idx)} predicted-C3 done "
          f"(+{(time.time()-t0)/60:.1f} min this run)", flush=True)
    return df_g


if __name__ == "__main__":
    dm = run_gender(1.0, "Male", "model_male.pkl", "val_male.pkl")
    dfem = run_gender(2.0, "Female", "model_female.pkl", "val_female.pkl")
    out = pd.concat([dm, dfem], ignore_index=True)
    out.to_csv(f"{TAB}/cohort.csv", index=False)
    print("WROTE", f"{TAB}/cohort.csv", "rows", len(out),
          "patients", out.pid.nunique(), flush=True)
