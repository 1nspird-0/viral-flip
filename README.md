# ViralFlip

**Predictive Illness Onset Forecasting Using Phone Sensor Data**

⚠️ **IMPORTANT SAFETY DISCLAIMER**: This is NOT a medical diagnosis device. ViralFlip provides early warning risk scores with uncertainty estimates for research and education purposes only. Do not use for clinical decision-making.

## Overview

ViralFlip predicts illness onset (e.g., flu-like symptoms, URI) at 24/48/72-hour horizons using passive and active phone sensor data:

- **Voice features**: F0, jitter, shimmer, HNR, MFCCs
- **Cough events**: Counts, burstiness, temporal patterns
- **Tapping test**: Motor control via inter-tap intervals
- **Gait analysis**: Cadence, smoothness, regularity
- **rPPG (camera)**: Heart rate, HRV proxies
- **GPS mobility**: Radius of gyration, location entropy
- **Ambient sensors**: Light patterns, barometric pressure
- **Screen events**: Sleep proxies, activity patterns

### Key Features

1. **Personal Baseline Memory (PBM)**: Within-person normalization using robust statistics
2. **Behavior-Drift Debiasing (BDD)**: Remove confounds from mobility/routine changes
3. **Drift-Lattice Hazard Network (DLHN)**: Multi-horizon prediction with lag structure
4. **Interpretable explanations**: Per-feature contribution breakdowns with counterfactuals
5. **Privacy-preserving**: Only derived features stored, no raw audio/video/GPS traces

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/viralflip.git
cd viralflip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- See `pyproject.toml` for full dependencies

## Quick Start

### Option A: Maximum Accuracy Training (GPU Recommended)

For RTX 3060+ or cloud GPU (Vast.ai, Lambda, etc.):

```bash
# One-command full pipeline (downloads, generates data, trains)
python scripts/train_max_accuracy.py

# Or step-by-step with high-performance config:
python scripts/download_more_data.py --full --parallel 4
python scripts/make_synthetic_episodes.py --max-accuracy
python scripts/train.py --config configs/high_performance.yaml --max-accuracy
```

### Option B: Quick Test (CPU/Low-Memory GPU)

```bash
# Generate synthetic data
python scripts/make_synthetic_episodes.py \
    --output data/synthetic \
    --n-users 100 \
    --days 90 \
    --seed 42
```

### Option C: Train with Real Public Data (Recommended)

```bash
# 1. Download all openly available datasets
python scripts/gather_real_data.py --output data/real --all

# 2. Prepare training data
python scripts/prepare_training_data.py --input data/real/processed --output data/training

# 3. View what's available
cat data/training/TRAINING_SUMMARY.txt
```

### 2. Train Model

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data synthetic \
    --output runs/my_experiment
```

### 3. Evaluate Model

```bash
python scripts/evaluate.py \
    --run_dir runs/my_experiment
```

### 4. Run Ablation Studies

```bash
python scripts/run_ablations.py \
    --run_dir runs/my_experiment
```

## Repository Structure

```
viralflip/
├── configs/
│   └── default.yaml          # Default configuration
├── data/
│   └── dataset_registry.yaml # Public dataset URLs
├── src/viralflip/
│   ├── features/             # Feature extraction modules
│   │   ├── voice.py          # Voice acoustic features
│   │   ├── cough.py          # Cough event features
│   │   ├── tapping.py        # Tapping test features
│   │   ├── gait_active.py    # Active gait features
│   │   ├── imu_passive.py    # Passive IMU features
│   │   ├── rppg.py           # Camera-based heart rate
│   │   ├── gps.py            # Mobility features
│   │   ├── light.py          # Ambient light features
│   │   ├── baro.py           # Barometer features
│   │   ├── screen.py         # Screen event features
│   │   └── quality.py        # Quality assessment
│   ├── baseline/
│   │   └── pbm.py            # Personal Baseline Memory
│   ├── debias/
│   │   └── ridge.py          # Behavior-Drift Debiasing
│   ├── model/
│   │   ├── drift_score.py    # Drift score compression
│   │   ├── lag_lattice.py    # Multi-horizon hazard model
│   │   ├── interactions.py   # Sparse interactions
│   │   ├── personalization.py # Per-user calibration
│   │   └── viralflip_model.py # Main model
│   ├── train/
│   │   ├── build_sequences.py # Sequence construction
│   │   ├── losses.py         # Loss functions
│   │   └── trainer.py        # Training loop
│   ├── eval/
│   │   ├── metrics.py        # AUPRC, AUROC, lead-time
│   │   ├── calibration.py    # ECE, Brier score
│   │   └── ablations.py      # Ablation runner
│   ├── explain/
│   │   └── explain.py        # Explanation engine
│   └── utils/
│       ├── io.py             # I/O utilities
│       ├── seed.py           # Reproducibility
│       └── logging.py        # Logging utilities
├── scripts/
│   ├── gather_real_data.py        # Download public datasets
│   ├── prepare_training_data.py   # Prepare data for training
│   ├── download_public_datasets.py # Dataset downloader (legacy)
│   ├── make_synthetic_episodes.py # Generate synthetic data
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── run_ablations.py           # Ablation studies
├── tests/
│   ├── test_pbm_no_leakage.py
│   ├── test_debias_ridge.py
│   ├── test_lag_lattice_shapes.py
│   └── test_explanations_counterfactual.py
├── pyproject.toml
└── README.md
```

## Configuration

Key configuration options in `configs/default.yaml`:

```yaml
data:
  bin_hours: 6              # Time bin duration
  horizons: [24, 48, 72]    # Prediction horizons (hours)

pbm:
  alpha: 0.03               # EMA update rate for mean
  safe_risk_threshold: 0.30 # Max risk for baseline update

bdd:
  ridge_lambda: 1.0         # Ridge regularization

model:
  max_lag_bins: 12          # Maximum lag (12 * 6h = 72h)
  use_interactions: false   # Sparse interactions (optional)

training:
  epochs: 100
  learning_rate: 0.001
  use_focal_loss: true
  pos_weight_multiplier: 5.0
```

## Evaluation Metrics

- **AUPRC**: Area Under Precision-Recall Curve (primary)
- **AUROC**: Area Under ROC Curve
- **Lead Time**: Fraction of episodes with early warning
- **False Alarms/Week**: Operational metric
- **ECE/Brier**: Calibration metrics

## Mandatory Ablations

The codebase includes ablation studies to verify component importance:

| Ablation | Description |
|----------|-------------|
| No PBM | Population-only normalization |
| No BDD | Skip behavior debiasing |
| No Voice | Remove voice modality |
| No Cough | Remove cough modality |
| No rPPG | Remove camera heart rate |
| No Gait+Tap | Remove motor features |
| No GPS | Remove mobility features |
| No Lag | Only use current timestep |
| No Personalization | Skip per-user calibration |

## Gathering Real Training Data

ViralFlip includes comprehensive scripts for downloading and processing public datasets:

```bash
# List all available datasets
python scripts/gather_real_data.py --list

# Download all open-access datasets (~3GB total)
python scripts/gather_real_data.py --output data/real --all

# Download by category
python scripts/gather_real_data.py --output data/real --category cough_voice
python scripts/gather_real_data.py --output data/real --category har_imu
python scripts/gather_real_data.py --output data/real --category phone_sensing

# Download specific datasets
python scripts/gather_real_data.py --output data/real --datasets coughvid,uci_har,esc50

# Prepare for training
python scripts/prepare_training_data.py --input data/real/processed --output data/training
```

### Available Datasets

| Dataset | Category | Access | Size | Use For |
|---------|----------|--------|------|---------|
| COUGHVID | cough_voice | Open | ~5GB | Cough detection |
| Coswara | cough_voice | Open | ~2GB | Cough/voice analysis |
| ESC-50 | cough_voice | Open | ~600MB | Audio classification |
| UCI HAR | har_imu | Open | ~60MB | IMU baseline |
| WISDM | har_imu | Open | ~400MB | Multi-sensor HAR |
| RealWorld HAR | har_imu | Open | ~2.5GB | Multi-modal |
| ExtraSensory | phone_sensing | Open | ~1GB | Behavior modeling |
| Beiwe Sample | phone_sensing | Open | ~100MB | Phone sensing |
| StudentLife | phone_sensing | Registration | ~1GB | Longitudinal behavior |
| UBFC-rPPG | rppg | Request | ~2GB | HR validation |

See `data/dataset_registry.yaml` for full URLs and access instructions.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pbm_no_leakage.py -v

# Run with coverage
pytest tests/ --cov=src/viralflip --cov-report=html
```

## Citation

If you use ViralFlip in your research, please cite:

```bibtex
@software{viralflip2024,
  title={ViralFlip: Predictive Illness Onset Forecasting Using Phone Sensor Data},
  year={2024},
  url={https://github.com/your-org/viralflip}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project builds on research in mobile health sensing, digital biomarkers, and interpretable machine learning. We thank the creators of the public datasets listed in the registry.

