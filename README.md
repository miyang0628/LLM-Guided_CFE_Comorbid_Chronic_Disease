# Pre-injection vs. post-hoc recourse under admissibility guardrails

Guardrail-constrained counterfactual recourse for comorbid chronic-disease
risk. Given a policyholder classified into a high-risk tier by a
gender-stratified gradient-boosting model, the framework searches for feasible
profile changes that move the policyholder toward a lower-risk tier while
remaining within physiologically admissible bounds.

The central methodological question is **where admissibility constraints must
enter the search**. We compare imposing the admissible region *before* the
counterfactual search against filtering unconstrained candidates *afterwards*,
holding the admissibility guarantee identical. Pre-injection preserves recourse;
post-hoc filtering destroys it. A three-layer guardrail architecture makes the
injection point explicit and separates non-negotiable safety floors (enforced in
code) from discretionary bands (set by rule or delegated to a language model).

---

## Headline results

Feasibility is the fraction of high-risk patients for whom at least one
admissible recourse candidate is returned.

| Constraint-handling scheme | Confirmed cohort (n=155) | Full predicted cohort (n=363) |
|---|---:|---:|
| C0 — unconstrained search | 100% (but ~100% inadmissible) | 100% |
| C1 — post-hoc filter, no fallback | **0.0%** | 0.0% |
| C1b — post-hoc filter, with fallback | 0.6% | ~0% |
| C2a — pre-injection, no fallback | 95.5% | ~95% |
| C2 — pre-injection, with fallback | **99.4%** | 99.2% |
| Rejection-sampling oracle (region non-empty) | 98.1% | — |

Three findings anchor the study:

1. **The injection point, not the fallback, drives feasibility.** Varying the
   injection point (post-hoc vs. pre-injection) and the stepwise target-class
   fallback independently, under a matched search budget, shows post-hoc
   filtering stays near 0% with or without fallback and pre-injection stays
   near ceiling with or without it.

2. **The collapse is mechanistic, not an under-powered or empty search.**
   Post-hoc feasibility does not improve as the candidate budget grows; an
   optimiser-free rejection-sampling oracle finds an admissible, correctly
   reclassified point for ~98% of patients; and the same pattern reproduces
   under a second, non-DiCE optimiser. Unconstrained candidates leave the
   admissible region *directionally* (wrong-sign moves and broken anthropometric
   coupling), not by marginal overshoot, so widening the bands does not rescue
   post-hoc filtering.

3. **Delegating the discretionary bands to a language model does not reproduce
   the rule.** Across three model tiers the LLM ranges diverge substantially
   from the fixed-ratio rule (mean interval IoU ≈ 0.4; every tier's 95%
   bootstrap CI far from the identity value 1, with large effect sizes),
   non-monotonically in model capability, and no tier removes the need for the
   code-enforced safety floor.

---

## The three-layer guardrail

1. **Layer 1 — absolute safety floors (code).** Non-negotiable nutritional and
   physiological floors, enforced in code and never delegated.
2. **Layer 2 — discretionary bands (rule *or* LLM).** Per-patient admissible
   ranges for modifiable features, including the structural anthropometric
   coupling (BMI = weight / height², with waist tracking weight at constant
   height).
3. **Layer 3 — constrained search.** The counterfactual optimiser (DiCE genetic
   search) draws candidates. Whether Layer 2 is imposed *before* or *after* this
   search is the injection-point question the study answers.

---

## Repository layout

```
notebooks/
  01_data_preprocessing.ipynb        KNHANES 2020–2024 cleaning & feature build
  02_xgboost_modeling.ipynb          gender-stratified gradient-boosting models
  03_representative_cases.ipynb      cohort selection, representative patients
  04_condition_comparison.ipynb      C0/C1/C2 constraint-handling comparison
  05_llm_vs_rule_guardrails.ipynb    LLM tier experiment (IoU, safety breaches)
  06_actuarial_projection.ipynb      cost-gradient sizing (illustrative)
  07_injection_and_uncertainty.ipynb injection×fallback factorial, feasibility
                                     oracle, budget curve, tier bootstrap CIs

  guardrail_core.py                  three-layer guardrail logic + IoU metrics
  experiment_core.py                 baseline C0–C2 evaluators, DiCE wrappers
  experiment_core_v2.py              injection×fallback factorial evaluator
                                     (matched budget), rejection-sampling
                                     oracle, candidate violation taxonomy
  run_cohort.py                      full predicted-Class-3 cohort driver
  build_tables.py                    renders the LaTeX result tables
  run_feasibility_curve.py           feasibility vs. search-budget sweep
  run_llm_uncertainty.py             bootstrap CIs + effect sizes for the tiers

data/                                df_final.pkl, model_{male,female}.pkl,
                                     val_{male,female}.pkl, agent_config.pkl
results/
  tables/                            result CSVs and rendered .tex tables
  figures/                           study figures
manuscript/                          main.tex, references.bib, figures, tables
```

## Quickstart

The DiCE and XGBoost versions matter for reproducibility:

```bash
pip install "xgboost==2.0.3" "dice-ml==0.11"
```

Run the pipeline end to end (from `notebooks/`, in order 01 → 06):

```bash
jupyter nbconvert --to notebook --execute 01_data_preprocessing.ipynb
# ... 02 → 03 → 04 → 05 → 06
```

The injection, feasibility-oracle, budget-curve and tier-uncertainty analyses
are gathered in `07_injection_and_uncertainty.ipynb`, which loads the cached
result CSVs and renders every table and the feasibility-curve figure in seconds.
To regenerate those caches from the models and data:

```bash
cd notebooks
python run_cohort.py              # -> results/tables/cohort.csv (363 patients × 5 conditions)
python build_tables.py            # -> LaTeX result tables
python run_feasibility_curve.py   # -> results/tables/feasibility_curve.csv
python run_llm_uncertainty.py     # -> tier CI tables
```

## Experimental design

**Cohort.** Patients predicted into the high-risk (comorbid) tier on the
held-out validation set: 363 in total (182 male, 181 female), of which 155 are
also confirmed by their true label. Results are reported on both the full
predicted cohort and the confirmed subset; they are nearly identical, so the
finding does not depend on conditioning on correct predictions.

**Constraint-handling schemes.** C0 runs the counterfactual search
unconstrained. C1 filters the C0 candidates against the admissible region after
the fact. C2 injects the same region into the search before it runs. The
injection point and the stepwise target-class fallback are varied as an
orthogonal factorial under a matched search budget (identical candidate count,
retries, target schedule and random seed), so the two factors are separately
identified.

**Feasibility oracle.** For each patient we also draw uniform candidates inside
the admissible band and test whether any is classified into a target class — an
optimiser-free check that the admissible region is non-empty, independent of the
counterfactual optimiser.

**LLM tiers.** The Layer-2 bands can be delegated to a language model instead of
fixed by rule. Three model tiers are compared against the fixed-ratio rule on
interval intersection-over-union (how patient-specific the bands are), raw
safety-floor breaches before code correction (reliance on Layer 1), and recourse
feasibility, in a fully paired design. Tier differences are reported as
patient-level bootstrap confidence intervals (B = 10,000) and effect sizes.

## Data

Korea National Health and Nutrition Examination Survey (KNHANES) 2020–2024,
9,738 observations after cleaning. KNHANES is cross-sectional and carries no
claims linkage, so a model-predicted risk-tier transition is a within-classifier
statement rather than an observed longitudinal outcome; the repository therefore
makes no expected-cost or pricing claim, and treats any monetary sizing as an
illustrative order-of-magnitude exercise only. The raw survey files and the
trained model artefacts are distributed with the study's data release rather
than in this repository; the cached result CSVs and the executed notebooks
contain all numbers reported in the paper.

## Notes on the LLM experiments

The tier experiments call the OpenAI API (weak / mid / strong tiers). The
bootstrap-CI analysis (`run_llm_uncertainty.py`) re-uses cached per-patient
outputs and makes no new API calls; the model snapshots, prompts, temperature
and seeds used to generate the cached outputs are recorded alongside the tier
experiment.

## Citation

If you use this code, please cite the accompanying article (citation to be added
on publication).

## License

MIT.
