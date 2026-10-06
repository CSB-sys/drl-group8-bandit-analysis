"""Reproduce the numerical checks used in the Group 8 presentation.

The paper's aggregate values are transcribed from Tables 3-5 of Wolf (2025),
arXiv:2510.27001. Each transcribed value is additionally cross-checked against
the author's published simulation output in the Bandit Playground repository
(github.com/eelisee/bandit_playground, branch v1.0-SKILL2025).

This script does not rerun the paper's one-million-step, 100-run simulations.
It verifies the calculations and interpretations used in the presentation.

Usage:
    python verify_bandit_analysis.py
    python verify_bandit_analysis.py --section madhulatha
    python verify_bandit_analysis.py --csv verification_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


HORIZON = 1_000_000


@dataclass(frozen=True)
class Scenario:
    name: str
    p_suboptimal: float
    p_optimal: float
    ucb_regret: float
    ucb_tuned_regret: float
    etc_regret: float
    ucb_tuned_suboptimal_ratio: float

    @property
    def gap(self) -> float:
        return self.p_optimal - self.p_suboptimal


SCENARIOS = {
    "A": Scenario("A", 0.800, 0.900, 238.17, 30.67, 10.10, 0.00030669),
    "B": Scenario("B", 0.895, 0.900, 1_127.17, 212.21, 150.66, 0.04244215),
    "C": Scenario("C", 0.890, 0.895, 1_172.57, 226.09, 170.53, 0.04521790),
}


# Values read from the author's published CSV output. Each file's final row is
# the T = 1,000,000 checkpoint; the column used is "Average Regret".
# Path pattern: data/<algorithm>/<config>/average_results_<p1>_<p2>.csv
REPO_CROSSCHECK = [
    # (algorithm, scenario, paper value, repo file, repo value)
    ("ETC (m=100)", "A", 10.10, "ETC/exploration_rounds_100/average_results_800_900.csv", 10.095),
    ("UCB-Tuned", "A", 30.67, "UCB-Tuned/default/average_results_800_900.csv", 30.669),
    ("EUCBV", "A", 47.70, "EUCBV/rho_0_5/average_results_800_900.csv", 47.727),
    ("UCB", "A", 238.17, "UCB/default/average_results_800_900.csv", 238.175),
    ("UCB-Improved", "A", 32_500.11, "UCB-Improved/delta_1/average_results_800_900.csv", 32_500.106),
    ("ETC (m=10000)", "B", 150.66, "ETC/exploration_rounds_10000/average_results_895_900.csv", 150.6631),
    ("UCB-Tuned", "B", 212.21, "UCB-Tuned/default/average_results_895_900.csv", 212.2084),
    ("UCB", "B", 1_127.17, "UCB/default/average_results_895_900.csv", 1_127.1738),
    ("ETC (m=10000)", "C", 170.53, "ETC/exploration_rounds_10000/average_results_890_895.csv", 170.5346),
    ("UCB-Tuned", "C", 226.09, "UCB-Tuned/default/average_results_890_895.csv", 226.087),
    ("UCB", "C", 1_172.57, "UCB/default/average_results_890_895.csv", 1_172.5742),
]


def bernoulli_variance(probability: float) -> float:
    """Return the variance p(1-p) of a Bernoulli random variable."""
    return probability * (1.0 - probability)


def regret_from_ratio(gap: float, suboptimal_ratio: float) -> float:
    """Reconstruct expected regret from gap, ratio, and the paper horizon."""
    return gap * suboptimal_ratio * HORIZON


def percentage_reduction(baseline: float, candidate: float) -> float:
    """Return the percentage reduction from baseline to candidate."""
    return (baseline - candidate) / baseline * 100.0


def chandan_checks() -> list[tuple[str, float, str]]:
    """Problem difficulty from the gaps, and the regret identity."""
    scenario_a = SCENARIOS["A"]
    scenario_b = SCENARIOS["B"]
    scenario_c = SCENARIOS["C"]
    return [
        ("Scenario A gap", scenario_a.gap, "0.100"),
        ("Scenario B gap", scenario_b.gap, "0.005"),
        ("Scenario C gap", scenario_c.gap, "0.005"),
        ("A gap / B gap", scenario_a.gap / scenario_b.gap, "20.00 times harder"),
        (
            "B UCB-Tuned regret from gap x suboptimal pulls",
            regret_from_ratio(scenario_b.gap, scenario_b.ucb_tuned_suboptimal_ratio),
            "212.21 (paper Table 4: 212.21)",
        ),
        (
            "C UCB-Tuned regret from gap x suboptimal pulls",
            regret_from_ratio(scenario_c.gap, scenario_c.ucb_tuned_suboptimal_ratio),
            "226.09 (paper Table 5: 226.09)",
        ),
    ]


def divakaran_checks() -> list[tuple[str, float, str]]:
    """Bernoulli variances, and the confidence-bound terms the code computes."""
    return [
        ("Bernoulli variance p=0.900", bernoulli_variance(0.900), "0.090000"),
        ("Bernoulli variance p=0.895", bernoulli_variance(0.895), "0.093975"),
        ("Bernoulli variance p=0.890", bernoulli_variance(0.890), "0.097900"),
        ("Bernoulli variance p=0.800", bernoulli_variance(0.800), "0.160000"),
        (
            "UCB-Tuned variance clip ceiling (UCB_Tuned.py line 51)",
            0.25,
            "min(1/4, V_ks) caps the variance term at 0.25",
        ),
        (
            "Max Bernoulli variance over tested arms",
            max(bernoulli_variance(p) for p in (0.800, 0.890, 0.895, 0.900)),
            "0.160000 - below the 0.25 clip, so the clip never binds here",
        ),
    ]


def madhulatha_checks() -> list[tuple[str, float, str]]:
    """Regret comparisons recomputed from the reported table values."""
    rows: list[tuple[str, float, str]] = []
    for name in ("A", "B", "C"):
        scenario = SCENARIOS[name]
        reduction = percentage_reduction(scenario.ucb_regret, scenario.ucb_tuned_regret)
        rows.append(
            (
                f"Scenario {name}: UCB-Tuned reduction vs UCB",
                reduction,
                f"{reduction:.2f}%",
            )
        )
    for name in ("B", "C"):
        scenario = SCENARIOS[name]
        advantage = percentage_reduction(scenario.ucb_tuned_regret, scenario.etc_regret)
        rows.append(
            (
                f"Scenario {name}: tuned ETC advantage over UCB-Tuned",
                advantage,
                f"{advantage:.2f}%",
            )
        )
    return rows


def akash_checks() -> list[tuple[str, float, str]]:
    """Whether the experiments isolate variance as the causal driver."""
    b_suboptimal = bernoulli_variance(0.895)
    b_optimal = bernoulli_variance(0.900)
    c_suboptimal = bernoulli_variance(0.890)
    c_optimal = bernoulli_variance(0.895)
    return [
        ("Scenario B variance at p=0.895", b_suboptimal, "0.093975"),
        ("Scenario B variance at p=0.900", b_optimal, "0.090000"),
        ("Scenario B variance difference", b_suboptimal - b_optimal, "0.003975"),
        ("Scenario C variance at p=0.890", c_suboptimal, "0.097900"),
        ("Scenario C variance at p=0.895", c_optimal, "0.093975"),
        ("Scenario C variance difference", c_suboptimal - c_optimal, "0.003925"),
        (
            "Paper baseline reward variance, p=0.895 at T=10^6 (Section 4.3)",
            HORIZON * bernoulli_variance(0.895),
            "93975 - matches the 93,975 printed in the paper",
        ),
    ]


def repo_checks() -> list[tuple[str, float, str]]:
    """Each paper value against the author's published simulation output."""
    rows: list[tuple[str, float, str]] = []
    for algorithm, scenario, paper_value, repo_file, repo_value in REPO_CROSSCHECK:
        rows.append(
            (
                f"{algorithm}, Scenario {scenario} [{repo_file}]",
                repo_value,
                f"paper {paper_value:,.2f} vs repo {repo_value:,.4f} - match",
            )
        )
    return rows


SECTIONS: dict[str, Callable[[], list[tuple[str, float, str]]]] = {
    "chandan": chandan_checks,
    "divakaran": divakaran_checks,
    "madhulatha": madhulatha_checks,
    "akash": akash_checks,
    "repo": repo_checks,
}


def validate_facts() -> None:
    """Fail explicitly if a transcribed value or formula check drifts."""
    # Scenario gaps
    assert math.isclose(SCENARIOS["A"].gap, 0.100, abs_tol=1e-12)
    assert math.isclose(SCENARIOS["B"].gap, 0.005, abs_tol=1e-12)
    assert math.isclose(SCENARIOS["C"].gap, 0.005, abs_tol=1e-12)
    assert math.isclose(SCENARIOS["A"].gap / SCENARIOS["B"].gap, 20.0, abs_tol=1e-9)

    # Bernoulli variances
    assert math.isclose(bernoulli_variance(0.900), 0.090000, abs_tol=1e-12)
    assert math.isclose(bernoulli_variance(0.895), 0.093975, abs_tol=1e-12)
    assert math.isclose(bernoulli_variance(0.890), 0.097900, abs_tol=1e-12)

    # The paper prints a baseline reward variance of 93,975 in Section 4.3.
    assert math.isclose(HORIZON * bernoulli_variance(0.895), 93_975.0, abs_tol=1e-6)

    # Regret reconstructed from gap x suboptimal pulls matches the tables.
    assert math.isclose(
        regret_from_ratio(SCENARIOS["B"].gap, SCENARIOS["B"].ucb_tuned_suboptimal_ratio),
        212.21,
        abs_tol=0.01,
    )
    assert math.isclose(
        regret_from_ratio(SCENARIOS["C"].gap, SCENARIOS["C"].ucb_tuned_suboptimal_ratio),
        226.09,
        abs_tol=0.01,
    )

    # Percentage reductions used on the results slides.
    assert math.isclose(percentage_reduction(1_127.17, 212.21), 81.1732036871102, abs_tol=1e-9)
    assert math.isclose(percentage_reduction(1_172.57, 226.09), 80.71842192790196, abs_tol=1e-9)

    # Every transcribed table value agrees with the author's published CSV output
    # to within rounding at the precision the paper reports.
    for algorithm, scenario, paper_value, repo_file, repo_value in REPO_CROSSCHECK:
        assert math.isclose(paper_value, repo_value, rel_tol=1e-3), (
            f"{algorithm} scenario {scenario}: paper {paper_value} vs repo {repo_value}"
        )


def selected_sections(section: str) -> list[tuple[str, list[tuple[str, float, str]]]]:
    if section == "all":
        return [(name, function()) for name, function in SECTIONS.items()]
    return [(section, SECTIONS[section]())]


def print_results(sections: list[tuple[str, list[tuple[str, float, str]]]]) -> None:
    for name, rows in sections:
        print(f"\n=== {name.upper()} VERIFICATION ===")
        for label, value, interpretation in rows:
            print(f"{label}: {value:.6f} -> {interpretation}")


def write_csv(
    output_path: Path,
    sections: list[tuple[str, list[tuple[str, float, str]]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["contributor", "calculation", "raw_value", "display_check"])
        for contributor, rows in sections:
            for label, value, interpretation in rows:
                writer.writerow([contributor, label, f"{value:.12f}", interpretation])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Group 8 bandit calculations from Wolf (2025), Tables 3-5."
    )
    parser.add_argument(
        "--section",
        choices=["all", *SECTIONS],
        default="all",
        help="Run all checks or one contributor's checks.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("verification_results.csv"),
        help="Path for the machine-readable CSV of the checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_facts()
    sections = selected_sections(args.section)
    print_results(sections)
    write_csv(args.csv, sections)
    print(f"\nCSV written to: {args.csv}")
    print("\nAll verification assertions passed.")
    print("Scope: calculation verification against paper Tables 3-5 and the")
    print("author's published CSV output. Not a rerun of the full simulation.")


if __name__ == "__main__":
    main()
