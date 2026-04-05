#!/usr/bin/env python3
"""
Results Summarization Script
==============================

Usage:
    python summarize_results.py

Aggregates all metrics from results/metrics/ into a final comparison
table and generates publication-style figures.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.plots import plot_model_comparison_bar, plot_per_class_f1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("summarize")


def main():
    parser = argparse.ArgumentParser(description="Summarize ML/DL pipeline results.")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Path to results directory (default: ./results).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_dir.exists():
        logger.error("No results found. Run the ML, DL, and robustness pipelines first.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 1. Aggregate test-set metrics across all models
    # ----------------------------------------------------------------
    test_metrics_files = sorted(metrics_dir.glob("*_test_metrics.csv"))
    if not test_metrics_files:
        logger.warning("No test metrics files found.")

    all_test_metrics = []
    for f in test_metrics_files:
        df = pd.read_csv(f)
        all_test_metrics.append(df)

    if all_test_metrics:
        test_summary = pd.concat(all_test_metrics, ignore_index=True)
        # Sort by accuracy descending
        test_summary = test_summary.sort_values("accuracy", ascending=False)
        summary_path = metrics_dir / "overall_test_summary.csv"
        test_summary.to_csv(summary_path, index=False)

        logger.info("\n" + "=" * 60)
        logger.info("OVERALL TEST SET PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"\n{test_summary.to_string(index=False)}")

        # Plot model comparison bar chart
        plot_model_comparison_bar(test_summary, plots_dir / "model_comparison.png")

    # ----------------------------------------------------------------
    # 2. Aggregate per-class F1 scores for comparison
    # ----------------------------------------------------------------
    report_files = sorted(metrics_dir.glob("*_test_report.csv"))
    per_class_results = {}

    for f in report_files:
        model_name = f.stem.replace("_test_report", "")
        df = pd.read_csv(f, index_col=0)
        # Extract per-class F1 (exclude summary rows)
        exclude_rows = {"accuracy", "macro avg", "weighted avg"}
        per_class = {}
        for idx in df.index:
            if idx not in exclude_rows:
                per_class[idx] = df.loc[idx, "f1-score"]
        if per_class:
            per_class_results[model_name] = per_class

    if per_class_results:
        # Use class names from the first result
        first_model = list(per_class_results.keys())[0]
        class_names = list(per_class_results[first_model].keys())
        plot_per_class_f1(per_class_results, class_names,
                          plots_dir / "per_class_f1_comparison.png")

    # ----------------------------------------------------------------
    # 3. Robustness summary
    # ----------------------------------------------------------------
    robustness_file = metrics_dir / "robustness_results.csv"
    if robustness_file.exists():
        rob_df = pd.read_csv(robustness_file)
        logger.info("\n" + "=" * 60)
        logger.info("ROBUSTNESS ANALYSIS RESULTS")
        logger.info("=" * 60)
        logger.info(f"\n{rob_df.to_string(index=False)}")

        # Pivot table for cleaner view
        pivot = rob_df.pivot_table(index="model", columns="condition",
                                    values="accuracy", aggfunc="first")
        pivot_path = metrics_dir / "robustness_pivot.csv"
        pivot.to_csv(pivot_path)
        logger.info(f"\nPivot table saved to {pivot_path}")
        logger.info(f"\n{pivot.to_string()}")
    else:
        logger.info("No robustness results found. Skipping.")

    # ----------------------------------------------------------------
    # 4. ML & DL summaries
    # ----------------------------------------------------------------
    ml_summary = metrics_dir / "ml_summary.csv"
    if ml_summary.exists():
        logger.info("\n" + "=" * 60)
        logger.info("ML PIPELINE SUMMARY")
        logger.info("=" * 60)
        ml_df = pd.read_csv(ml_summary)
        logger.info(f"\n{ml_df.to_string(index=False)}")

    dl_summary = metrics_dir / "dl_summary.csv"
    if dl_summary.exists():
        logger.info("\n" + "=" * 60)
        logger.info("DL PIPELINE SUMMARY")
        logger.info("=" * 60)
        dl_df = pd.read_csv(dl_summary)
        logger.info(f"\n{dl_df.to_string(index=False)}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Summarization Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"All plots saved to {plots_dir}/")
    logger.info(f"All metrics saved to {metrics_dir}/")


if __name__ == "__main__":
    main()
