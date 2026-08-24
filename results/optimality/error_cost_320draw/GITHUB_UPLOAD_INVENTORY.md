# Error-cost sensitivity analysis files

Code:
- hddm/06_error_cost_pilot.py
- hddm/07_error_cost_320draw.py

Formal result directory:
- results/optimality/error_cost_320draw/

Key files:
- error_cost_draws_final.csv
- error_cost_summary_final.csv
- error_cost_probabilities_final.csv
- error_cost_validation_final.json
- error_cost_run_config.json
- error_cost_input_manifest.json
- frozen_inputs/selected_posterior_draws.csv
- frozen_inputs/reconstructed_reward_criteria.csv

Checkpoint retained for audit/resume:
- error_cost_draws_checkpoint.csv

Primary sensitivity objective:
U(a;q) = [10*P(correct & RT<criterion) - q_points*P(wrong)] / E[T_trial]

q_relative values:
0.0, 0.5, 1.0, 2.0, 3.0, 4.0
