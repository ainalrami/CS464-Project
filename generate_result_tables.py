#!/usr/bin/env python3
"""
Generate markdown tables from results CSV files.

Usage:
    python generate_result_tables.py [--results-dir ./results]

Prints all tables in markdown format so they can be copied into the report.
Also saves each table as a separate CSV in results/metrics/ for easy access.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def csv_to_markdown(csv_path, title, round_digits=4):
    path = Path(csv_path)
    if not path.exists():
        print(f"\n⚠️  Not found (run pipeline first): {path}")
        return None
    df = pd.read_csv(path)
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].round(round_digits)
    print(f"\n### {title}")
    print(f"*Source: `{path}`*\n")
    print(df.to_markdown(index=False))
    print()
    return df


def per_class_table(report_csv, title):
    path = Path(report_csv)
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    exclude = {"accuracy", "macro avg", "weighted avg"}
    df = df[[c for c in df.columns if c != "support" or True]]  # keep all cols
    df = df[~df.index.isin(exclude)]
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].round(4)
    print(f"\n### {title}")
    print(f"*Source: `{path}`*\n")
    print(df.to_markdown())
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics = results_dir / "metrics"

    if not metrics.exists():
        print("❌ No results yet. Run the pipelines first:")
        print("   python run_ml.py --config configs/ml.yaml")
        print("   python run_dl.py --config configs/dl.yaml")
        print("   python run_robustness.py --config configs/robustness.yaml")
        print("   python summarize_results.py")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("RESULT TABLES — Markdown Format")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Table 1: All models ranked by test accuracy
    # ------------------------------------------------------------------
    csv_to_markdown(
        metrics / "overall_test_summary.csv",
        "Table 1: Overall Test Set Performance (All Models, Ranked by Accuracy)",
    )

    # ------------------------------------------------------------------
    # Table 2: ML pipeline summary (all feature modes + sizes)
    # ------------------------------------------------------------------
    csv_to_markdown(
        metrics / "ml_summary.csv",
        "Table 2: Classical ML Pipeline Summary",
    )

    # ------------------------------------------------------------------
    # Table 3: Upscaling comparison
    # ------------------------------------------------------------------
    csv_to_markdown(
        metrics / "upscaling_comparison.csv",
        "Table 3: Upscaling Experiment — Native 64×64 vs Upscaled 128×128",
    )

    # ------------------------------------------------------------------
    # Table 4: DL pipeline summary
    # ------------------------------------------------------------------
    csv_to_markdown(
        metrics / "dl_summary.csv",
        "Table 4: Deep Learning Pipeline Summary",
    )

    # ------------------------------------------------------------------
    # Table 5: Robustness pivot (models × conditions)
    # ------------------------------------------------------------------
    csv_to_markdown(
        metrics / "robustness_pivot.csv",
        "Table 5: Robustness Evaluation — Accuracy by Degradation Condition",
    )

    # ------------------------------------------------------------------
    # Table 6: Robustness full results
    # ------------------------------------------------------------------
    csv_to_markdown(
        metrics / "robustness_results.csv",
        "Table 6: Robustness Full Results",
    )

    # ------------------------------------------------------------------
    # Per-class tables for every model that has a test report
    # ------------------------------------------------------------------
    report_files = sorted(metrics.glob("*_test_report.csv"))
    if report_files:
        print("\n" + "=" * 80)
        print("PER-CLASS PERFORMANCE TABLES")
        print("=" * 80)
        for f in report_files:
            model_name = f.stem.replace("_test_report", "")
            per_class_table(f, f"Per-Class Results — {model_name} (Test Set)")

    print("=" * 80)
    print("✅  Done — copy the tables above into your report.")
    print("=" * 80)


if __name__ == "__main__":
    main()
