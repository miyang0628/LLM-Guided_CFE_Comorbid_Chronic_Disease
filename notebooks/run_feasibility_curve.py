"""
run_feasibility_curve.py
========================
Feasibility curve over search budget, plus a rejection-sampling
lower bound. For a range of per-patient candidate budgets (total_CFs), measure
post-hoc-filter feasibility vs pre-injection feasibility on the confirmed
Class-3 cohort. If post-hoc feasibility stays ~0 as the budget grows while the
rejection-sampling oracle finds admissible points, the collapse is about the
injection point, not an under-powered search.

Runs on the confirmed cohort (true & predicted Class 3). Uses a fixed seed and
identical fallback settings across the two schemes at every budget.

Output: results/tables/feasibility_curve.csv
"""
import os, time, warnings, joblib
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
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
df_final = joblib.load(f"{DATA}/df_final.pkl")
ac = joblib.load(f"{DATA}/agent_config.pkl")
X, T, VARY = ac["X_features"], ac["target_col"], ac["vary_features"]

BUDGETS = [2, 4, 8, 16]
SEED = 0
SUBSAMPLE = 40   # patients per gender; enough to trace the curve shape


def run_gender(code, name, model_file, val_file):
    model = joblib.load(f"{DATA}/{model_file}")
    val = joblib.load(f"{DATA}/{val_file}")
    df_stable = df_final[df_final["Sex"] == code].copy().astype(float)
    pred = model.predict(val["X_val"])
    idx = val["X_val"][(val["y_val"] == 3) & (pred == 3)].index.tolist()
    rng = np.random.default_rng(SEED)
    if len(idx) > SUBSAMPLE:
        idx = sorted(rng.choice(idx, size=SUBSAMPLE, replace=False).tolist())
    exp = ec2.make_dice(model, df_stable, X, T)
    ckpt = f"{TAB}/_fc_{name.lower()}.csv"
    rows = []; done = set()
    if os.path.exists(ckpt):
        prev = pd.read_csv(ckpt)
        # a patient is done if it has all schemes x budgets rows
        cnt = prev.groupby("pid").size()
        expected = 4 + len(BUDGETS) * 2  # 4 rejection budgets + posthoc+pre per budget
        done = set(cnt[cnt >= expected].index.tolist())
        rows = prev[prev.pid.isin(done)].to_dict("records")
        print(f"{name}: resuming FC, {len(done)} done", flush=True)
    t0 = time.time()
    todo = [p for p in idx if p not in done]
    for k, pid in enumerate(todo):
        q = val["X_val"].loc[[pid]][X]
        cur = {f: float(q.iloc[0][f]) for f in X}
        g = gc.build_rule_guardrails(cur, X, "aggressive")
        # rejection-sampling oracle at increasing sample budgets
        for ns in [100, 500, 2000, 10000]:
            rs = ec2.rejection_sampling_baseline(model, q, cur, X, VARY, g,
                                                 n_samples=ns, seed=SEED)
            rows.append({"pid": pid, "gender": name, "scheme": "rejection",
                         "budget": ns, "feasible": rs["rs_feasible"],
                         "hit_rate": rs["rs_hit_rate"]})
        # DiCE post-hoc vs pre-injection at increasing candidate budgets
        for B in BUDGETS:
            rp = ec2.evaluate(exp, q, df_stable, X, VARY, cur,
                              injection="post", fallback=True,
                              total_CFs=B, tries=3, seed=SEED)
            rows.append({"pid": pid, "gender": name, "scheme": "posthoc",
                         "budget": B, "feasible": rp["feasible"],
                         "hit_rate": np.nan})
            ri = ec2.evaluate(exp, q, df_stable, X, VARY, cur,
                              injection="pre", fallback=True, guardrails=g,
                              total_CFs=B, tries=3, seed=SEED)
            rows.append({"pid": pid, "gender": name, "scheme": "preinjection",
                         "budget": B, "feasible": ri["feasible"],
                         "hit_rate": np.nan})
        if (k + 1) % 5 == 0:
            pd.DataFrame(rows).to_csv(ckpt, index=False)
            print(f"  {name} {len(done)+k+1}/{len(idx)} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ckpt, index=False)
    print(f"{name} done {(time.time()-t0)/60:.1f} min", flush=True)
    return d


if __name__ == "__main__":
    dm = run_gender(1.0, "Male", "model_male.pkl", "val_male.pkl")
    dfem = run_gender(2.0, "Female", "model_female.pkl", "val_female.pkl")
    out = pd.concat([dm, dfem], ignore_index=True)
    out.to_csv(f"{TAB}/feasibility_curve.csv", index=False)
    # summary
    piv = (out.groupby(["scheme", "budget"])["feasible"].mean()
              .reset_index())
    print(piv.to_string())
    print("WROTE feasibility_curve.csv", flush=True)
