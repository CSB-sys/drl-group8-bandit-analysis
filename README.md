# Group 8 — Verification Work for Wolf (2025)

Supporting evidence for BITS Pilani WILP, Deep Reinforcement Learning, Assignment Problem I.

**Paper analysed:** Elise Wolf (2025), *A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms*, [arXiv:2510.27001](https://arxiv.org/abs/2510.27001)

**Author's codebase:** [github.com/eelisee/bandit_playground](https://github.com/eelisee/bandit_playground/tree/v1.0-SKILL2025), branch `v1.0-SKILL2025`

## Scope of this repository

This repository contains **our verification of the paper**, not an implementation of it. The algorithms and simulation data belong to the paper's author.

What we did:

1. Transcribed every result quoted in our presentation from the paper's Tables 3–5.
2. Cross-checked each transcribed value against the author's published simulation output (the CSV files in her repository).
3. Checked the confidence-bound formulas on our slides against her source code, line by line.
4. Recomputed every derived figure — percentage reductions, variance differences, the regret identity — ourselves.

What we did **not** do: re-run the paper's full simulation. That is 100 independent trials of 1,000,000 steps across 8 algorithm families. Our arithmetic is verified; the underlying simulation is the author's.

## Verification result: 11 of 11 values match

Each file below is from the author's repository under `data/`. The final row of each file is the T = 1,000,000 checkpoint, and the column used is `Average Regret`. File names encode the arm probabilities: `800_900` is Scenario A (p₁ = 0.800, p₂ = 0.900), `895_900` is Scenario B, `890_895` is Scenario C.

| Algorithm | Scenario | Paper value | Author's CSV | Match |
|---|---|---:|---:|:---:|
| ETC (m=100) | A | 10.10 | 10.0950 | ✅ |
| UCB-Tuned | A | 30.67 | 30.6690 | ✅ |
| EUCBV | A | 47.70 | 47.7270 | ✅ |
| UCB | A | 238.17 | 238.1750 | ✅ |
| UCB-Improved | A | 32,500.11 | 32,500.1060 | ✅ |
| ETC (m=10,000) | B | 150.66 | 150.6631 | ✅ |
| UCB-Tuned | B | 212.21 | 212.2084 | ✅ |
| UCB | B | 1,127.17 | 1,127.1738 | ✅ |
| ETC (m=10,000) | C | 170.53 | 170.5346 | ✅ |
| UCB-Tuned | C | 226.09 | 226.0870 | ✅ |
| UCB | C | 1,172.57 | 1,172.5742 | ✅ |

## Formula verification against the author's source code

| Claim on our slides | Verified in |
|---|---|
| Plain UCB bonus is √(2 log t / T(a)) — no variance term | `src/algorithms/UCB.py`, line 27 |
| UCB-Tuned adds an empirical variance term, clipped at 1/4 | `src/algorithms/UCB_Tuned.py`, lines 50–51 |
| UCB-V adds √(2σ̂²ε/T) + 3bε/T where ε = θ log t | `src/algorithms/UCB_V.py`, line 65 |
| Deterministic seeding, horizon T = 10⁶ | `src/config.py` (`global_seed = 42`, `time_horizons`) |

## Derived figures we recomputed

| Figure | Value | How |
|---|---:|---|
| Difficulty ratio, Scenario A vs B/C | 20× | 0.100 / 0.005 |
| UCB-Tuned regret reduction vs UCB, Scenario B | 81.17% | (1127.17 − 212.21) / 1127.17 |
| UCB-Tuned regret reduction vs UCB, Scenario C | 80.72% | (1172.57 − 226.09) / 1172.57 |
| Tuned ETC advantage over UCB-Tuned, Scenario B | 29.00% | (212.21 − 150.66) / 212.21 |
| Bernoulli variance difference, Scenario B | 0.003975 | 0.895(1−0.895) − 0.900(1−0.900) |
| Bernoulli variance difference, Scenario C | 0.003925 | 0.890(1−0.890) − 0.895(1−0.895) |
| Baseline reward variance, p = 0.895 at T = 10⁶ | 93,975 | 10⁶ × 0.895 × 0.105 — matches the figure printed in the paper's §4.3 |

The regret identity also holds: regret ≈ gap × suboptimal pulls. For UCB-Tuned in Scenario B, 0.005 × 42,442 = 212.21, matching Table 4 exactly.

## Our main critical finding

For Bernoulli rewards, variance is p(1−p), so the mean and the variance are **mathematically coupled** — changing p changes both. Scenarios B and C differ in their probability pairs, so they differ in mean *and* variance simultaneously.

The experiments therefore show that variance-aware methods perform well in these settings, but they do **not** isolate variance as the causal driver. A cleaner test would hold the means fixed while varying variance independently, which requires a reward family richer than Bernoulli.

## Contents

```
verification/
  verify_bandit_analysis.py              Dependency-free verification script
  Group8_Bandit_Analysis_Verification.ipynb   Same checks as a Colab notebook
  verification_results.csv               Generated output
  output.txt                             Console output from a clean run
handwritten/                             Each contributor's rough working
slides/                                  Final presentation
```

Run it with:

```bash
python verification/verify_bandit_analysis.py
```

No dependencies beyond the Python standard library.

## Contributors

| Name | BITS ID | Verification work |
|---|---|---|
| Chandan Singh Baghel | 2025AG05120 | Suboptimality gaps, difficulty ratio, regret identity cross-check |
| Divakaran Ramu | 2025AG05828 | Confidence-bound formulas against source code, experimental setup against `config.py` |
| Kurella Madhulatha | 2025AG05826 | Results extraction from Tables 3–5, CSV cross-check, percentage recomputation |
| Akash Choudhury | 2025AG05368 | Bernoulli variance coupling, causal-validity critique, §4.3 variance figure |
