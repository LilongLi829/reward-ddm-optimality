# Reward-related decision threshold and reward-rate optimality analysis

This repository contains the reproducible analysis code for a hierarchical
Drift Diffusion Model (HDDM) analysis of reward-related changes in decision
threshold and subsequent reward-rate optimality analyses.

The analysis is based on Experiment 2 of the Braeutigam Stroop dataset.

## Analysis overview

The analysis consists of the following steps:

1. Prepare and validate HDDM input data.
2. Fit candidate hierarchical drift diffusion models.
3. Run four independent MCMC chains for each candidate model.
4. Assess MCMC convergence.
5. Perform posterior predictive checks (PPC).
6. Compare candidate models.
7. Extract posterior estimates of reward and no-reward decision thresholds.
8. Reconstruct the experimental reward criterion.
9. Estimate the reward-rate optimal decision threshold.
10. Propagate posterior uncertainty using 320 posterior draws.

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
        └── posterior_uncertainty_320draw/
            ├── posterior_optimality_draws_final.csv
            ├── posterior_optimality_probabilities_final.csv
            ├── posterior_optimality_summary_final.csv
            ├── reconstructed_reward_criteria.csv
            └── selected_posterior_draws.csv
