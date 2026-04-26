"""
Master script to run all experiments sequentially.
Usage: conda run -n nih_research python run_all.py

Experiments:
  0. Synthetic synergy (interaction model)
  1. Consistent MI estimation (BRCA + GBMLGG)
  1b. Separate-classifier MI estimation (BRCA, for Table 3)
  2. VMIB compression-prediction tradeoff
  3. Missing-modality robustness (v2)
  4. Fusion collapse diagnostics (v2)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_experiment(name, module_name):
    print(f"\n{'#' * 80}")
    print(f"# STARTING: {name}")
    print(f"{'#' * 80}\n")
    start = time.time()

    module = __import__(module_name)
    module.main()

    elapsed = time.time() - start
    print(f"\n>>> {name} completed in {elapsed:.1f}s\n")
    return elapsed


if __name__ == "__main__":
    total_start = time.time()

    times = {}
    experiments = [
        ("Exp 0: Synthetic Synergy", "exp0_synthetic_synergy"),
        ("Exp 1: Information Decomposition", "exp1_information_decomposition"),
        ("Exp 1c: Consistent MI Estimation", "exp1_consistent_mi"),
        ("Exp 2: VMIB Compression-Prediction Tradeoff", "exp2_vmib_tradeoff"),
        ("Exp 3v2: Missing-Modality Robustness", "exp3_missing_modality_v2"),
        ("Exp 4v2: Fusion Collapse Diagnostics", "exp4_fusion_collapse_v2"),
    ]

    for name, module in experiments:
        times[name] = run_experiment(name, module)

    total = time.time() - total_start
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    for name, t in times.items():
        print(f"  {name}: {t:.1f}s")
    print(f"  TOTAL: {total:.1f}s")
