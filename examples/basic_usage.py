#!/usr/bin/env python3
"""
Basic usage example for SALUS-SafetyQA Benchmark

This script demonstrates how to:
1. Load the benchmark dataset
2. Examine question structure
3. Calculate basic statistics
4. Filter questions by category
"""

import json
from collections import Counter


def load_benchmark(benchmark_path: str = "../benchmark/SALUS_SafetyQA_v1.0.json"):
    """Load the benchmark dataset."""
    with open(benchmark_path, "r") as f:
        return json.load(f)


def print_question(question: dict):
    """Pretty print a single question."""
    print(f"\n{'=' * 80}")
    print(f"Question Type: {question['question_type']}")
    print(f"Source Type: {question['source_type']}")
    if question["country"]:
        location = (
            f"{question['province']}, {question['country']}"
            if question["province"]
            else question["country"]
        )
        print(f"Location: {location}")
    print(f"\n{question['mc_question']}")
    print("\nOptions:")
    for opt in question["mc_options"]:
        marker = "✓" if opt["is_correct"] else " "
        print(f"  {marker} {opt['label']}) {opt['text']}")
    print(f"\nCorrect Answer: {question['mc_correct_answer']}")
    print(f"Expected Answer: {question['expected_answer']}")
    print(f"{'=' * 80}\n")


def analyze_dataset(questions: list):
    """Print dataset statistics."""
    print(f"\n{'=' * 80}")
    print("DATASET STATISTICS")
    print(f"{'=' * 80}\n")

    print(f"Total Questions: {len(questions)}")

    # Question types
    print("\nQuestion Type Distribution:")
    q_types = Counter(q["question_type"] for q in questions)
    for qtype, count in q_types.most_common():
        pct = (count / len(questions)) * 100
        print(f"  {qtype:20s}: {count:4d} ({pct:5.1f}%)")

    # Source types
    print("\nSource Type Distribution:")
    s_types = Counter(q["source_type"] for q in questions)
    for stype, count in s_types.most_common():
        pct = (count / len(questions)) * 100
        print(f"  {stype:20s}: {count:4d} ({pct:5.1f}%)")

    # Jurisdictions
    jurisdictional = [q for q in questions if q["country"] is not None]
    print(
        f"\nJurisdictional Questions: {len(jurisdictional)} ({len(jurisdictional) / len(questions) * 100:.1f}%)"
    )

    if jurisdictional:
        countries = Counter(q["country"] for q in jurisdictional)
        print("  By Country:")
        for country, count in countries.most_common():
            print(f"    {country}: {count}")


def filter_questions(questions: list, **filters):
    """Filter questions by criteria.

    Examples:
        filter_questions(questions, question_type='specification')
        filter_questions(questions, source_type='SDS')
        filter_questions(questions, country='US', province='MI')
    """
    filtered = questions
    for key, value in filters.items():
        filtered = [q for q in filtered if q.get(key) == value]
    return filtered


def main():
    """Main example script."""
    # Load benchmark
    print("Loading SALUS-SafetyQA Benchmark...")
    questions = load_benchmark()

    # Print overall statistics
    analyze_dataset(questions)

    # Show example questions from different categories
    print(f"\n{'=' * 80}")
    print("EXAMPLE QUESTIONS")
    print(f"{'=' * 80}")

    # Example 1: Specification question
    spec_questions = filter_questions(questions, question_type="specification")
    if spec_questions:
        print("\n1. Example SPECIFICATION question:")
        print_question(spec_questions[0])

    # Example 2: Compliance question from regulation
    compliance_reg = filter_questions(
        questions, question_type="compliance", source_type="REGULATION"
    )
    if compliance_reg:
        print("\n2. Example COMPLIANCE question from REGULATION:")
        print_question(compliance_reg[0])

    # Example 3: PPE question
    ppe_questions = filter_questions(questions, question_type="what_ppe")
    if ppe_questions:
        print("\n3. Example WHAT_PPE question:")
        print_question(ppe_questions[0])

    # Example 4: Jurisdictional question (Michigan)
    mi_questions = filter_questions(questions, country="US", province="MI")
    if mi_questions:
        print("\n4. Example MICHIGAN-specific question:")
        print_question(mi_questions[0])

    # Show how to iterate and evaluate
    print(f"\n{'=' * 80}")
    print("SIMPLE EVALUATION LOOP")
    print(f"{'=' * 80}\n")

    print("Example code to evaluate your model:")
    print("""
    def evaluate_model(questions, model_predict_fn):
        correct = 0
        for q in questions:
            # Your model predicts 'a', 'b', 'c', or 'd'
            prediction = model_predict_fn(q['mc_question'], q['mc_options'])

            if prediction == q['mc_correct_answer']:
                correct += 1

        accuracy = correct / len(questions)
        return accuracy
    """)


if __name__ == "__main__":
    main()
