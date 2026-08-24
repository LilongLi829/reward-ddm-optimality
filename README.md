# Reward-related decision threshold and reward-rate optimality analysis

This repository contains the analysis code and main output files for a hierarchical Drift Diffusion Model (HDDM) analysis of reward-related changes in decision threshold, followed by a reward-rate optimality analysis and a posterior sensitivity analysis that adds a prespecified subjective cost of wrong responses.

The analysis is based on the Stroop task from Experiment 2 of Bräutigam et al. (2024).

## Analysis overview

The analysis consists of the following steps:

1. Prepare and validate the HDDM input data.
2. Fit candidate hierarchical drift diffusion models.
3. Run four independent MCMC chains for each candidate model.
4. Assess MCMC convergence.
5. Perform posterior predictive checks (PPC).
6. Compare candidate models.
7. Estimate reward and no-reward decision thresholds.
8. Reconstruct the experimental reward criterion.
9. Estimate the reward-rate optimal decision threshold.
10. Propagate posterior uncertainty using 320 posterior draws.
11. Run a group-level pilot to explore a broader non-correct/error-cost range.
12. Repeat the formal wrong-response-cost sensitivity analysis over the same frozen 320 joint posterior draws.

The main question is whether participants adjust their decision threshold under reward toward the threshold predicted to maximize reward rate.

## Repository structure

```text
reward-ddm-optimality/
├── README.md
├── .gitignore
├── .gitattributes
│
├── hddm/
│   ├── 00_check_hddm_input.py
│   ├── 01_fit_model.py
│   ├── 01_fit_model_standalone.py
│   ├── 02_check_convergence_v2.py
│   ├── 03_ppc_v2.py
│   ├── 04_reward_rate_optimality.py
│   ├── 05_final_optimality_320draw.py
│   ├── 06_error_cost_pilot.py
│   ├── 07_error_cost_320draw.py
│   ├── model_factory.py
│   └── run_all_formal.sh
│
├── data/
│   └── processed/
│       └── Braeutigam_Exp2_Stroop_HDDM.csv
│
└── results/
    ├── diagnostics/
    │   ├── convergence_report.txt
    │   ├── dic_by_chain.csv
    │   ├── rhat_all.csv
    │   └── rhat_summary.csv
    │
    ├── ppc/
    │   ├── all_models_condition_ppc_comparison.csv
    │   ├── all_models_effect_ppc_comparison.csv
    │   └── m_va/
    │       ├── condition_ppc_comparison.csv
    │       ├── effect_ppc_comparison.csv
    │       ├── observed_behavioral_effects.csv
    │       └── observed_condition_summary.csv
    │
    ├── thresholds/
    │   ├── m_va_threshold_posterior.csv
    │   ├── m_va_threshold_probability.csv
    │   └── m_va_threshold_summary.csv
    │
    └── optimality/
        ├── reconstructed_reward_criteria.csv
        ├── reward_ddm_group_parameters.csv
        ├── reward_rate_curve_group.csv
        ├── reward_rate_curves.png
        ├── reward_rate_optimality_summary.csv
        ├── posterior_uncertainty_320draw/
            ├── posterior_optimality_draws_final.csv
            ├── posterior_optimality_probabilities_final.csv
            ├── posterior_optimality_summary_final.csv
            ├── reconstructed_reward_criteria.csv
            └── selected_posterior_draws.csv
        └── error_cost_320draw/
            ├── error_cost_draws_final.csv
            ├── error_cost_summary_final.csv
            ├── error_cost_probabilities_final.csv
            ├── error_cost_validation_final.json
            ├── error_cost_run_config.json
            ├── error_cost_input_manifest.json
            ├── error_cost_draws_checkpoint.csv
            ├── GITHUB_UPLOAD_INVENTORY.md
            └── frozen_inputs/
                ├── reconstructed_reward_criteria.csv
                └── selected_posterior_draws.csv
```

## Data

### Processed HDDM data

The processed trial-level data used for HDDM fitting are included in this repository:

```text
data/processed/Braeutigam_Exp2_Stroop_HDDM.csv
```

After the predefined exclusions and HDDM coding procedure, the final HDDM dataset contains:

```text
Participants: 37
Valid trials: 10,651
```

### Original raw data

The original raw data are **not redistributed in this repository**.

The preregistrations and raw data for all experiments reported by Bräutigam et al. (2024) are publicly available from the authors' Open Science Framework (OSF) project:

```text
https://osf.io/dbeq2/
```

For the reward-rate optimality analyses, download the Experiment 2 raw data file and create the following local directory:

```text
data/raw/
```

Place the raw data file at:

```text
data/raw/RawData_Exp2.txt
```

The resulting local data structure should therefore be:

```text
data/
├── raw/
│   └── RawData_Exp2.txt
└── processed/
    └── Braeutigam_Exp2_Stroop_HDDM.csv
```

`RawData_Exp2.txt` is required by the reward-criterion reconstruction and reward-rate optimality scripts.

## Computational environment

The reported HDDM analyses were run inside a Docker/JupyterLab environment.

The environment used for the analyses was:

```text
Python 3.12.11
HDDM 1.0.1RC
PyMC 2.3.8
```

The analysis scripts assume that the project is available inside the Docker container at:

```text
/home/jovyan/project
```

Therefore, the recommended setup is to place or mount the repository at this location before running the scripts.

Because HDDM relies on a specialized software environment, running the analyses inside a compatible Docker environment is recommended.

## Running the analysis

The following commands assume that the working directory inside the Docker container is:

```text
/home/jovyan/project
```

Before running the reward-rate analyses, make sure that:

```text
data/raw/RawData_Exp2.txt
```

has been downloaded from the original OSF dataset and placed in the expected location.

### 1. Validate the HDDM input data

```bash
python hddm/00_check_hddm_input.py
```

This script checks the processed HDDM input data before model fitting.

### 2. Fit the candidate HDDM models

The formal model-fitting procedure can be started with:

```bash
bash hddm/run_all_formal.sh
```

Three candidate models are compared:

```text
mv:  v ~ Reward × Congruency; a = constant

ma:  v ~ Congruency; a ~ Reward

mva: v ~ Reward × Congruency; a ~ Reward
```

For each candidate model, four independent MCMC chains are run.

The formal sampling settings are:

```text
Iterations per chain: 5000
Burn-in:              2500
Thinning:             2
Number of chains:     4
```

### 3. Check convergence and compare models

```bash
python hddm/02_check_convergence_v2.py
```

This step evaluates convergence across the four chains using R-hat and summarizes DIC values for model comparison.

### 4. Posterior predictive checks

```bash
python hddm/03_ppc_v2.py
```

Posterior predictive checks evaluate whether the fitted models reproduce the main behavioral characteristics of the four Reward × Congruency conditions, including accuracy and response-time measures.

### 5. Reward-rate optimality analysis

```bash
python hddm/04_reward_rate_optimality.py
```

This analysis reconstructs the block-specific reward RT criterion from the original Experiment 2 data and calculates expected reward rate across candidate decision thresholds.

The main analysis follows the intended experimental reward rule:

* the response must be correct;
* the response must be faster than the block-specific reward criterion.

For a candidate decision threshold (a), expected reward rate is defined as:

```text
RR(a) =
10 × P(correct and RT < reward criterion | a)
------------------------------------------------
            expected trial duration
```

Expected trial duration includes the reward cue, fixation interval, response time, feedback duration, and inter-trial interval.

The theoretical optimal threshold is the value of (a) that maximizes expected reward rate.

### 6. Propagate posterior uncertainty

```bash
python hddm/05_final_optimality_320draw.py
```

The final analysis propagates uncertainty in the HDDM parameters into the optimality calculation using 320 balanced posterior draws from the four formal MCMC chains.

For each posterior draw, the analysis estimates:

* the observed reward-condition decision threshold;
* the theoretically optimal threshold;
* the gap between the observed and optimal thresholds;
* reward-rate efficiency.

A sensitivity analysis is also conducted using the reward rule reconstructed from the experimental log.

### 7. Pilot the subjective error-cost extension

```bash
python hddm/06_error_cost_pilot.py
```

This group-level pilot was used only to determine a sensible range for the formal sensitivity analysis. In this exploratory script, the pilot penalty is applied to all non-correct trials (`1 - accuracy`), whereas the formal Step 8 analysis separates wrong responses from timeouts and applies the additional cost to wrong responses only. The pilot is not treated as an independent fitted psychological model.

### 8. Formal 320-draw subjective error-cost sensitivity analysis

```bash
python hddm/07_error_cost_320draw.py
```

The formal sensitivity analysis reuses the same frozen 320 joint posterior draws as Step 6 and does **not** refit HDDM. Within each posterior draw, `v_RI`, `v_RC`, `t`, `z = 0.5`, and the experimental timing/reward structure are held fixed while the decision threshold `a` is varied.

The extended utility-rate objective is:

```text
U(a; q) =
10 × P(correct and RT < reward criterion | a)
- q_points × P(wrong response | a)
------------------------------------------------
              expected trial duration
```

where `q_points = 10 × q_relative`. The prespecified sensitivity values are:

```text
q_relative = 0, 0.5, 1, 2, 3, 4
q_points   = 0, 5, 10, 20, 30, 40
```

`q_relative` is a **sensitivity-analysis weight**, not a psychological parameter estimated from the present data. The additional cost is applied to wrong responses only; timeouts are tracked separately. Reward-incongruent and reward-congruent conditions are simulated separately and combined with equal weighting.

A three-draw smoke test can be run first:

```bash
python hddm/07_error_cost_320draw.py --max-new-draws 3
```

and then resumed with:

```bash
python hddm/07_error_cost_320draw.py --resume
```

The script saves a checkpoint after each completed posterior draw.

## Main HDDM model

Model comparison favored the `mva` model:

```text
v ~ Reward × Congruency
a ~ Reward
```

This model allows reward and congruency to influence evidence accumulation and allows the decision threshold to vary between reward and no-reward conditions.

## Main results

The estimated group-level decision thresholds were approximately:

```text
No-reward condition: a_NR = 1.261
Reward condition:    a_R  = 1.134
```

Approximately 97% of posterior samples supported a lower decision threshold under reward.

Thus, participants generally became less cautious when reward was available.

However, under the intended reward rule, the estimated reward-rate optimal threshold was:

```text
a* ≈ 0.527
```

The observed reward-condition threshold therefore remained substantially higher than the threshold predicted by pure reward-rate maximization.

The difference between the observed reward threshold and the theoretical optimum was approximately:

```text
a_R - a* ≈ 0.605
```

The observed reward-condition strategy achieved approximately:

```text
71.5%
```

of the theoretically maximal reward rate.

Thus, reward shifted participants' decision thresholds in the direction predicted by reward-rate maximization, but the adjustment was not large enough to reach the theoretical optimum.

## Subjective error-cost sensitivity results

Adding a prespecified subjective cost for wrong responses shifted the theoretical optimum upward toward the observed reward-condition threshold:

| `q_relative` | mean `a*(q)` | mean `a_R - a*(q)` | `P[a_R > a*(q)]` |
|---:|---:|---:|---:|
| 0.0 | 0.527 | 0.605 | 1.000 |
| 0.5 | 0.618 | 0.514 | 1.000 |
| 1.0 | 0.709 | 0.423 | 1.000 |
| 2.0 | 0.894 | 0.238 | 1.000 |
| 3.0 | 1.060 | 0.072 | 0.875 |
| 4.0 | 1.202 | -0.070 | 0.191 |

For `q_relative <= 2`, all 320 posterior draws still had `a_R > a*(q)`. At `q_relative = 3`, the mean gap was approximately `0.072` with a 95% credible interval of `[-0.051, 0.212]`; at `q_relative = 4`, the mean gap was approximately `-0.070` with a 95% credible interval of `[-0.239, 0.078]`.

These results show that the conclusion depends on the objective function: under the pure objective reward-rate benchmark participants remained more cautious than optimal, whereas sufficiently strong additional wrong-response costs can move the normative optimum close to or above the observed threshold. The analysis does **not** establish that participants' true subjective error cost equals any particular tested value of `q_relative`.

The `q_relative = 0` condition exactly reproduces the original intended-payoff analysis: mean `a* = 0.5270625` across 320 draws, with a maximum absolute difference from the original per-draw optimum of approximately `2.8e-16`.

## Reward-rule sensitivity analysis

Reconstruction of the experimental log indicated that a small number of sufficiently fast error responses were also recorded as rewarded.

For this reason, the main analysis follows the intended reward rule described in the experiment:

```text
correct + sufficiently fast → reward
```

An additional sensitivity analysis follows the reward rule observed in the experimental log:

```text
sufficiently fast → reward
```

Under the logged-payoff rule, the estimated optimal threshold was lower than under the intended-payoff rule.

However, both analyses led to the same qualitative conclusion: the observed reward-condition decision threshold remained substantially higher than the corresponding reward-rate optimum.

## Output files

### Model diagnostics

```text
results/diagnostics/
```

contains convergence summaries, R-hat statistics, and DIC results.

### Posterior predictive checks

```text
results/ppc/
```

contains observed and posterior-predictive behavioral summaries.

### Decision-threshold estimates

```text
results/thresholds/
```

contains posterior summaries of the reward and no-reward decision thresholds used in the accompanying report.

### Reward-rate optimality

```text
results/optimality/
```

contains the reconstructed reward criterion, reward-rate curves, group-level optimality results, posterior uncertainty analyses, the reward-rule sensitivity analysis, and the formal 320-draw subjective error-cost sensitivity analysis. The main error-cost results are stored under `results/optimality/error_cost_320draw/`.

## Reproducibility notes

The result files included in this repository correspond to the formal analyses reported in the accompanying report.

HDDM fitting relies on MCMC sampling. Therefore, complete re-fitting of the models may produce small numerical differences because of Monte Carlo variability. The main inferential patterns should nevertheless remain stable when the same data, model specifications, and computational environment are used.

The reward-rate optimality analysis also relies on stochastic simulation. Posterior uncertainty is therefore propagated across 320 posterior draws rather than relying only on a single point estimate. The formal subjective error-cost analysis reuses these same frozen draws and common-random-number logic. Its `q_relative = 0` condition provides a direct reproduction check against the original intended-payoff analysis.

The original raw experimental data should be obtained from the authors' OSF repository rather than redistributed through this repository.

## Reference

Bräutigam, L. C., Leuthold, H., Mackenzie, I. G., & Mittelstädt, V. (2024). Proactive reward in conflict tasks: Does it only enhance general performance or also modulate conflict effects? *Attention, Perception, & Psychophysics, 86*, 2153–2168.

DOI:

```text
https://doi.org/10.3758/s13414-024-02896-5
```

Original OSF project:

```text
https://osf.io/dbeq2/
```
