#!/usr/bin/env python3
"""One-command maximum accuracy training for ViralFlip.

This script runs the full pipeline optimized for your RTX 5070:
1. Downloads all datasets (with resume support)
2. Generates large-scale synthetic data
3. Trains with high-performance config
4. Runs ablation studies
5. Outputs comprehensive metrics

Usage:
    python scripts/train_max_accuracy.py
    python scripts/train_max_accuracy.py --skip-download  # Skip data download
    python scripts/train_max_accuracy.py --resume         # Resume from last checkpoint
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n[OK] {description} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n[ERROR] {description} failed (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Full max-accuracy training pipeline")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip dataset download")
    parser.add_argument("--skip-synthetic", action="store_true",
                       help="Skip synthetic data generation (use existing)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: runs/max_accuracy_TIMESTAMP)")
    parser.add_argument("--parallel-downloads", type=int, default=4,
                       help="Number of parallel downloads")
    parser.add_argument("--skip-ablations", action="store_true",
                       help="Skip ablation studies after training")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/max_accuracy_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          VIRALFLIP MAXIMUM ACCURACY TRAINING                 ║
╠══════════════════════════════════════════════════════════════╣
║  Output: {str(output_dir):<50} ║
║  Config: configs/high_performance.yaml                       ║
║                                                              ║
║  Optimized for:                                              ║
║  - RTX 5070 (12GB VRAM)                                      ║
║  - Mixed precision (FP16)                                    ║
║  - Large batch (128 x 2 accum = 256 effective)              ║
║  - 500 users x 180 days synthetic data                       ║
║  - Cosine warmup LR schedule                                 ║
║  - Full interaction modeling                                 ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    start_time = time.time()
    
    # Step 1: Download datasets
    if not args.skip_download:
        success = run_command(
            [sys.executable, "scripts/download_more_data.py", 
             "--full", f"--parallel={args.parallel_downloads}"],
            "Downloading all datasets for maximum accuracy"
        )
        if not success:
            print("\nWarning: Some downloads may have failed. Continuing with available data...")
    else:
        print("\n[SKIP] Dataset download skipped")
    
    # Step 2: Generate synthetic data
    if not args.skip_synthetic:
        success = run_command(
            [sys.executable, "scripts/make_synthetic_episodes.py",
             "--max-accuracy", "--output", "data/synthetic_large"],
            "Generating large-scale synthetic data (500 users x 180 days)"
        )
        if not success:
            print("Failed to generate synthetic data")
            return 1
    else:
        print("\n[SKIP] Synthetic data generation skipped")
    
    # Step 3: Train model
    train_cmd = [
        sys.executable, "scripts/train.py",
        "--config", "configs/high_performance.yaml",
        "--data", "data/synthetic_large",
        "--output", str(output_dir),
    ]
    
    if args.resume:
        checkpoint = output_dir / "checkpoint.pt"
        if checkpoint.exists():
            train_cmd.extend(["--resume", str(checkpoint)])
    
    success = run_command(
        train_cmd,
        "Training with high-performance configuration"
    )
    
    if not success:
        print("Training failed")
        return 1
    
    # Step 4: Run ablation studies
    if not args.skip_ablations:
        success = run_command(
            [sys.executable, "scripts/run_ablations.py",
             "--run_dir", str(output_dir)],
            "Running ablation studies"
        )
        if not success:
            print("Warning: Ablation studies failed")
    
    # Step 5: Final evaluation
    success = run_command(
        [sys.executable, "scripts/evaluate.py",
         "--run_dir", str(output_dir)],
        "Final evaluation"
    )
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETE!                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total time: {hours:02d}h {minutes:02d}m                                          ║
║  Output: {str(output_dir):<50} ║
║                                                              ║
║  Key files:                                                  ║
║  - best_model.pt      : Best trained model                   ║
║  - training_history.pkl: Training curves                     ║
║  - config.yaml        : Configuration used                   ║
║  - train.log          : Full training log                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Print results summary
    try:
        import pickle
        history_path = output_dir / "training_history.pkl"
        if history_path.exists():
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            
            best_auprc = max(history.get('val_auprc', [0]))
            best_auroc = max(history.get('val_auroc', [0]))
            
            print(f"RESULTS:")
            print(f"  Best Validation AUPRC: {best_auprc:.4f}")
            print(f"  Best Validation AUROC: {best_auroc:.4f}")
    except Exception:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

