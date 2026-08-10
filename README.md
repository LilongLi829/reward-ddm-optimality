\# Reward-related decision threshold and reward-rate optimality analysis



This repository contains the reproducible analysis code for a hierarchical

Drift Diffusion Model (HDDM) analysis of reward-related changes in decision

threshold and subsequent reward-rate optimality analyses.



The analysis is based on Experiment 2 of the Braeutigam Stroop dataset.



\## Analysis overview



The analysis consists of the following steps:



1\. Prepare and validate HDDM input data.

2\. Fit candidate hierarchical drift diffusion models.

3\. Run four independent MCMC chains for each candidate model.

4\. Assess MCMC convergence.

5\. Perform posterior predictive checks (PPC).

6\. Compare candidate models.

7\. Extract posterior estimates of reward and no-reward decision thresholds.

8\. Reconstruct the experimental reward criterion.

9\. Estimate the reward-rate optimal decision threshold.

10\. Propagate posterior uncertainty using 320 posterior draws.



\## Repository structure



```text

reward-ddm-optimality/

├── README.md

├── hddm/

│   ├── 00\_check\_hddm\_input.py

│   ├── 01\_fit\_model.py

│   ├── 01\_fit\_model\_standalone.py

│   ├── 02\_check\_convergence\_v2.py

│   ├── 03\_ppc\_v2.py

│   ├── 04\_reward\_rate\_optimality.py

│   ├── 05\_final\_optimality\_320draw.py

│   ├── model\_factory.py

│   └── run\_all\_formal.sh

├── data/

│   └── processed/

├── results/

└── figures/

