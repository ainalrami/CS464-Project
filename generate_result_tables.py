#!/usr/bin/env python3
"""
Generate markdown tables from results CSV files for embedding in reports.

Usage:
    python generate_result_tables.py

This script converts CSV outputs to formatted markdown tables that can be
copied directly into a report.
"""

import pandas as pd
from pathlib import Path
import sys

def csv_to_markdown_table(csv_path, title):
    """Convert a CSV file to a markdown table."""
    if not Path(csv_path).exists():
        print(f"⚠️  File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Round numeric columns to 4 decimals for readability
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].round(4)
    
    print(f"\n{title}")
    print("=" * 80)
    print(f"Source: {csv_path}\n")
    print(df.to_markdown(index=False))
    print(f"\n")

def main():
    results_dir = Path("./results/metrics")
    
    if not results_dir.exists():
        print("❌ Results directory not found. Run the pipeline first:")
        print("   python run_ml.py --config configs/ml.yaml")
        print("   python run_dl.py --config configs/dl.yaml")
        print("   python run_robustness.py --config configs/robustness.yaml")
        print("   python summarize_results.py")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("RESULT TABLES (Markdown Format)")
    print("=" * 80)
    
    # Main comparison
    csv_to_markdown_table(
        results_dir / "overall_test_summary.csv",
        "Table 1: Overall Test Set Performance (All Models)"
    )
    
    # ML summary
    csv_to_markdown_table(
        results_dir / "ml_summary.csv",
        "Table 2: Classical ML Models Summary"
    )
    
    # DL summary
    csv_to_markdown_table(
        results_dir / "dl_summary.csv",
        "Table 3: Deep Learning Model Summary"
    )
    
    # Robustness results
    csv_to_markdown_table(
        results_dir / "robustness_results.csv",
        "Table 4: Robustness Evaluation Results"
    )
    
    # Robustness pivot
    csv_to_markdown_table(
        results_dir / "robustness_pivot.csv",
        "Table 5: Robustness Pivot (Models × Degradation Conditions)"
    )
    
    print("\n" + "=" * 80)
    print("✅ Tables generated successfully!")
    print("=" * 80)
    print("\n📋 Copy the markdown tables above directly into your report.")
    print("💡 Tip: Use http://markdowntable.com/ to convert to other formats if needed.\n")

if __name__ == "__main__":
    main()
