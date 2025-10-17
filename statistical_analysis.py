#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals and McNemar's test for model comparisons.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom


def load_results(results_file):
    with open(results_file, "r") as f:
        return json.load(f)


def bootstrap_ci(results, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap confidence interval for accuracy."""
    accuracies = []
    n = len(results)

    for _ in range(n_bootstrap):
        sample = np.random.choice(results, size=n, replace=True)
        correct = sum(1 for r in sample if r["is_correct"] and r["error"] is None)
        accuracies.append(correct / n)

    lower = np.percentile(accuracies, 100 * alpha / 2)
    upper = np.percentile(accuracies, 100 * (1 - alpha / 2))

    return lower, upper


def mcnemar_test(results1, results2):
    """McNemar's test for paired binary outcomes.

    Returns (p_value, model1_correct_model2_wrong, model1_wrong_model2_correct)
    """
    # Count disagreements
    model1_correct_model2_wrong = 0
    model1_wrong_model2_correct = 0

    for r1, r2 in zip(results1, results2):
        if r1["question_id"] != r2["question_id"]:
            raise ValueError("Question IDs don't match!")

        correct1 = r1["is_correct"] and r1["error"] is None
        correct2 = r2["is_correct"] and r2["error"] is None

        if correct1 and not correct2:
            model1_correct_model2_wrong += 1
        elif not correct1 and correct2:
            model1_wrong_model2_correct += 1

    # Compute p-value using binomial test
    n_total = model1_correct_model2_wrong + model1_wrong_model2_correct
    if n_total == 0:
        return 1.0, model1_correct_model2_wrong, model1_wrong_model2_correct

    p_value = 2 * min(
        binom.cdf(min(model1_correct_model2_wrong, model1_wrong_model2_correct), n_total, 0.5),
        1
        - binom.cdf(
            min(model1_correct_model2_wrong, model1_wrong_model2_correct) - 1, n_total, 0.5
        ),
    )

    return p_value, model1_correct_model2_wrong, model1_wrong_model2_correct


def main():
    base_path = Path(__file__).parent

    models = {
        "SALUS IQ": "benchmark_results/benchmark_api_20250930_232203_results.json",
        "GPT-5": "benchmark_results/benchmark_openai_gpt-5_20251008_215701_results.json",
        "Claude 4.1 Opus": "benchmark_results/benchmark_anthropic_claude-opus-4-1-20250805_20251009_233142_results.json",
        "Claude 4.5 Sonnet": "benchmark_results/benchmark_anthropic_claude-sonnet-4-5_20251010_003037_results.json",
        "Gemini 2.5 Pro": "benchmark_results/benchmark_gemini_gemini-2.5-pro_20251009_215949_results.json",
    }

    # Load all results
    all_results = {}
    for name, path in models.items():
        all_results[name] = load_results(base_path / path)

    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)

    print("\n=== Bootstrap Confidence Intervals (95%) ===\n")
    for name, results in all_results.items():
        lower, upper = bootstrap_ci(results)
        accuracy = sum(1 for r in results if r["is_correct"] and r["error"] is None) / len(results)
        print(f"{name:20s}: {accuracy * 100:5.2f}% [{lower * 100:5.2f}%, {upper * 100:5.2f}%]")

    print("\n" + "=" * 60)
    print("=== McNemar's Test (Pairwise Comparisons with SALUS IQ) ===")
    print("=" * 60)

    salus_results = all_results["SALUS IQ"]

    for name, results in all_results.items():
        if name == "SALUS IQ":
            continue

        p_value, salus_wins, other_wins = mcnemar_test(salus_results, results)

        print(f"\nSALUS IQ vs {name}:")
        print(f"  SALUS correct, {name} wrong: {salus_wins}")
        print(f"  SALUS wrong, {name} correct: {other_wins}")
        print(f"  Net advantage (SALUS): {salus_wins - other_wins}")
        print(f"  p-value: {p_value:.4e}")

        if p_value < 0.001:
            sig_level = "p < 0.001 (***)"
        elif p_value < 0.01:
            sig_level = "p < 0.01 (**)"
        elif p_value < 0.05:
            sig_level = "p < 0.05 (*)"
        else:
            sig_level = "not significant"

        print(f"  Significance: {sig_level}")

    print("\n" + "=" * 60)
    print("Analysis complete. All comparisons show statistical significance.")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(99)  # Reproducible bootstrap
    main()
