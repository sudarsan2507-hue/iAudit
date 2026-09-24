"""Master execution script for iAudit pipeline.
Runs all stages end-to-end with validation and progress tracking.
"""
import sys
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

STAGES = [
    ("Stage T2: Train Tabular Regressors (5-Fold CV)", "train_tabular.py"),
    ("Stage T3: Export Standardized Predictions", "write_tabular_predictions.py"),
    ("Stage 5: Full Fairness Audit & Statistical Testing", "fairness_audit.py"),
    ("Stage T4: Cross-Model Comparison & Bias Shape", "compare_models.py"),
    ("Stage V: IOP Scanner Feature Distribution Analysis", "check_iop_features.py"),
    ("Stage R: Unsupervised Reliability Score & Selective Prediction", "reliability_score.py"),
]

def run_stage(name, script_name):
    print("\n" + "=" * 75)
    print(f"[*] RUNNING: {name}")
    print(f"    Script: scripts/{script_name}")
    print("=" * 75)
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print("STDERR:", result.stderr.strip())
        
    if result.returncode != 0:
        print(f"\n[!] ERROR in {script_name} (exit code {result.returncode})")
        return False
        
    print(f"\n[+] SUCCESS: {name} completed in {elapsed:.2f}s")
    return True

def main():
    print("=" * 75)
    print(" iAudit: Brain-Age Fairness Audit - End-to-End Pipeline Execution")
    print(f" Repository: {REPO_ROOT}")
    print("=" * 75)
    
    total_start = time.time()
    for name, script in STAGES:
        success = run_stage(name, script)
        if not success:
            print(f"\n[!] Pipeline halted at {script}")
            sys.exit(1)
            
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"[+] ALL 6 PIPELINE STAGES COMPLETED SUCCESSFULLY in {total_elapsed:.2f}s")
    print("    Results, figures, and fairness reports are ready in results/")
    print("=" * 75)

if __name__ == "__main__":
    main()
