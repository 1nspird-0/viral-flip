#!/usr/bin/env python3
"""Train ViralFlip model.

Usage:
    python scripts/train.py --config configs/default.yaml --data synthetic
    python scripts/train.py --config configs/high_performance.yaml --data synthetic --max-accuracy
    python scripts/train.py --config configs/default.yaml --data data/my_dataset
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from viralflip.baseline.pbm import PersonalBaselineMemory
from viralflip.debias.ridge import BehaviorDriftDebiaser
from viralflip.model.viralflip_model import ViralFlipModel
from viralflip.train.build_sequences import SequenceBuilder, UserData, UserBin, UserDataset
from viralflip.train.trainer import ViralFlipTrainer
from viralflip.utils.io import load_config, load_pickle, save_pickle, ensure_dir
from viralflip.utils.logging import setup_logging, get_logger
from viralflip.utils.seed import set_seed

try:
    from viralflip.utils.gpu import setup_gpu, print_gpu_info, get_gpu_info
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False


logger = get_logger(__name__)


# Feature dimensions
FEATURE_DIMS = {
    "voice": 24,
    "cough": 6,
    "tap": 6,
    "gait_active": 8,
    "rppg": 5,
    "imu_passive": 6,
    "gps": 5,
    "light": 4,
    "baro": 4,
    "screen": 5,
}

PHYSIOLOGY_MODALITIES = ["voice", "cough", "tap", "gait_active", "rppg", "light", "baro"]


def load_synthetic_data(data_dir: Path) -> list[UserData]:
    """Load synthetic dataset.
    
    Args:
        data_dir: Directory containing synthetic data.
        
    Returns:
        List of UserData objects.
    """
    metadata_path = data_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    
    user_data_list = []
    
    for user_id in metadata["users"]:
        user_path = data_dir / f"{user_id}.pkl"
        if not user_path.exists():
            logger.warning(f"User data not found: {user_path}")
            continue
        
        raw = load_pickle(user_path)
        
        # Convert to UserData format
        bins = []
        for t in range(raw["n_bins"]):
            features = {m: raw["features"][m][t] for m in raw["features"]}
            quality = {m: np.array([raw["quality"][m][t]]) for m in raw["quality"]}
            missing = {m: raw["missing"][m][t] for m in raw["missing"]}
            labels = {h: int(raw["labels"][h][t]) for h in raw["labels"]}
            
            bins.append(UserBin(
                user_id=raw["user_id"],
                bin_idx=t,
                timestamp=float(t),
                features=features,
                quality=quality,
                missing=missing,
                labels=labels,
                in_washout=raw["washout"][t],
            ))
        
        user_data_list.append(UserData(
            user_id=raw["user_id"],
            bins=bins,
            onset_timestamps=[float(o) for o in raw["onset_indices"]],
        ))
    
    logger.info(f"Loaded {len(user_data_list)} users from {data_dir}")
    
    return user_data_list


def initialize_pbm(
    train_users: list[UserData],
    config: dict,
) -> PersonalBaselineMemory:
    """Initialize PBM from training data.
    
    Args:
        train_users: List of training user data.
        config: Configuration dict.
        
    Returns:
        Initialized PBM.
    """
    pbm_cfg = config.get("pbm", {})
    
    pbm = PersonalBaselineMemory(
        feature_dims=FEATURE_DIMS,
        alpha=pbm_cfg.get("alpha", 0.03),
        beta=pbm_cfg.get("beta", 0.02),
        safe_risk_threshold=pbm_cfg.get("safe_risk_threshold", 0.30),
        init_bins=pbm_cfg.get("init_bins", 56),
    )
    
    # Initialize from early healthy bins
    for user_data in train_users:
        for bin_data in user_data.bins[:56]:  # First 14 days
            if bin_data.in_washout:
                continue
            
            for modality in FEATURE_DIMS:
                if modality in bin_data.features and not bin_data.missing.get(modality, False):
                    quality = bin_data.quality.get(modality, np.array([1.0]))[0]
                    pbm.add_init_sample(
                        user_data.user_id,
                        modality,
                        bin_data.features[modality],
                        quality,
                    )
    
    logger.info(f"Initialized PBM for {len(train_users)} users")
    
    return pbm


def fit_debiaser(
    train_users: list[UserData],
    pbm: PersonalBaselineMemory,
    config: dict,
) -> BehaviorDriftDebiaser:
    """Fit behavior-drift debiaser on healthy data.
    
    Args:
        train_users: List of training user data.
        pbm: Personal baseline memory.
        config: Configuration dict.
        
    Returns:
        Fitted debiaser.
    """
    bdd_cfg = config.get("bdd", {})
    
    debiaser = BehaviorDriftDebiaser(
        feature_dims=FEATURE_DIMS,
        ridge_lambda=bdd_cfg.get("ridge_lambda", 1.0),
    )
    
    # Collect healthy drift data
    behavior_drifts = {m: [] for m in debiaser.behavior_blocks}
    physiology_drifts = {m: [] for m in debiaser.physiology_blocks}
    
    for user_data in train_users:
        for bin_data in user_data.bins:
            # Only use healthy bins (not in washout, no positive labels)
            if bin_data.in_washout:
                continue
            if any(bin_data.labels.get(h, 0) == 1 for h in [24, 48, 72]):
                continue
            
            # Compute drifts
            for modality in FEATURE_DIMS:
                if modality not in bin_data.features:
                    continue
                if bin_data.missing.get(modality, False):
                    continue
                
                drift = pbm.compute_drift(
                    user_data.user_id,
                    modality,
                    bin_data.features[modality],
                )
                
                if modality in debiaser.behavior_blocks:
                    behavior_drifts[modality].append(drift)
                elif modality in debiaser.physiology_blocks:
                    physiology_drifts[modality].append(drift)
    
    # Convert to arrays
    n_samples = min(
        len(behavior_drifts[m]) for m in debiaser.behavior_blocks
        if len(behavior_drifts[m]) > 0
    )
    
    if n_samples == 0:
        logger.warning("No healthy samples for debiasing")
        return debiaser
    
    behavior_arrays = {
        m: np.array(behavior_drifts[m][:n_samples])
        for m in debiaser.behavior_blocks
        if len(behavior_drifts[m]) > 0
    }
    
    physiology_arrays = {
        m: np.array(physiology_drifts[m][:n_samples])
        for m in debiaser.physiology_blocks
        if len(physiology_drifts[m]) > 0
    }
    
    # Fit debiaser
    debiaser.fit(behavior_arrays, physiology_arrays)
    
    logger.info(f"Fitted debiaser on {n_samples} healthy samples")
    
    return debiaser


def build_datasets(
    users: list[UserData],
    pbm: PersonalBaselineMemory,
    debiaser: BehaviorDriftDebiaser,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """Build train and validation sequences.
    
    Args:
        users: List of user data.
        pbm: Personal baseline memory.
        debiaser: Behavior drift debiaser.
        config: Configuration dict.
        
    Returns:
        Tuple of (train_sequences, val_sequences).
    """
    splits_cfg = config.get("splits", {})
    
    builder = SequenceBuilder(
        horizons=config.get("data", {}).get("horizons", [24, 48, 72]),
        bin_hours=config.get("data", {}).get("bin_hours", 6),
        max_lag=config.get("model", {}).get("max_lag_bins", 12),
        train_user_frac=splits_cfg.get("train_user_frac", 0.7),
        val_user_frac=splits_cfg.get("val_user_frac", 0.15),
        train_time_frac=splits_cfg.get("train_time_frac", 0.7),
        val_time_frac=splits_cfg.get("val_time_frac", 0.15),
        seed=config.get("training", {}).get("seed", 42),
    )
    
    # Split users
    user_ids = [u.user_id for u in users]
    train_ids, val_ids, test_ids = builder.split_users(user_ids)
    
    logger.info(f"User split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Build sequences with drift computation
    train_sequences = []
    val_sequences = []
    
    for user_data in users:
        # Determine time split
        time_train, time_val, time_test = builder.split_temporal(len(user_data.bins))
        
        # Compute drifts
        for bin_data in user_data.bins:
            # Compute raw drifts
            raw_drifts = {}
            for modality in FEATURE_DIMS:
                if modality in bin_data.features:
                    raw_drifts[modality] = pbm.compute_drift(
                        user_data.user_id,
                        modality,
                        bin_data.features[modality],
                    )
            
            # Apply debiasing
            if debiaser.is_fitted:
                behavior_drift = {m: raw_drifts[m] for m in debiaser.behavior_blocks if m in raw_drifts}
                physiology_drift = {m: raw_drifts[m] for m in debiaser.physiology_blocks if m in raw_drifts}
                
                debiased = debiaser.debias(behavior_drift, physiology_drift)
                
                # Update features with debiased drifts
                for m, d in debiased.items():
                    raw_drifts[m] = d
            
            # Store in bin_data
            bin_data.features = raw_drifts
        
        # Build sequences
        if user_data.user_id in train_ids:
            seqs = builder.build_sequences(user_data, time_train, list(FEATURE_DIMS.keys()))
            train_sequences.extend(seqs)
        elif user_data.user_id in val_ids:
            seqs = builder.build_sequences(user_data, time_val, list(FEATURE_DIMS.keys()))
            val_sequences.extend(seqs)
    
    logger.info(f"Built {len(train_sequences)} train, {len(val_sequences)} val sequences")
    
    return train_sequences, val_sequences


def main():
    parser = argparse.ArgumentParser(description="Train ViralFlip model")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--data", "-d", type=str, default="synthetic",
                       help="Data source: 'synthetic' or path to data directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output directory (default: runs/TIMESTAMP)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'auto', 'cpu', 'cuda'")
    parser.add_argument("--max-accuracy", action="store_true",
                       help="Use high-performance config for maximum accuracy")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")
    parser.add_argument("--gpu-info", action="store_true",
                       help="Print GPU info and exit")
    
    args = parser.parse_args()
    
    # Print GPU info if requested
    if args.gpu_info:
        if GPU_UTILS_AVAILABLE:
            print_gpu_info()
        else:
            print("GPU utilities not available")
        return
    
    # Override config for max accuracy mode
    if args.max_accuracy:
        args.config = "configs/high_performance.yaml"
        logger.info("Using high-performance configuration for maximum accuracy")
    
    # Load config
    config = load_config(args.config)
    
    # Setup output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.get("paths", {}).get("output_dir", "runs")) / timestamp
    else:
        output_dir = Path(args.output)
    
    output_dir = ensure_dir(output_dir)
    
    # Setup logging
    setup_logging(log_file=output_dir / "train.log")
    logger.info(f"Output directory: {output_dir}")
    
    # Set seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)
    
    # Load data
    if args.data == "synthetic":
        data_dir = Path(config.get("paths", {}).get("data_dir", "data")) / "synthetic"
        if not data_dir.exists():
            logger.info("Generating synthetic data...")
            from viralflip.scripts.make_synthetic_episodes import (
                SyntheticConfig, generate_synthetic_dataset
            )
            synth_cfg = config.get("synthetic", {})
            generate_synthetic_dataset(
                SyntheticConfig(**synth_cfg),
                data_dir,
            )
    else:
        data_dir = Path(args.data)
    
    users = load_synthetic_data(data_dir)
    
    # Split users first
    splits_cfg = config.get("splits", {})
    rng = np.random.default_rng(seed)
    user_ids = [u.user_id for u in users]
    shuffled = rng.permutation(user_ids)
    
    n_train = int(len(users) * splits_cfg.get("train_user_frac", 0.7))
    n_val = int(len(users) * splits_cfg.get("val_user_frac", 0.15))
    
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])
    
    train_users = [u for u in users if u.user_id in train_ids]
    val_users = [u for u in users if u.user_id in val_ids]
    
    # Initialize PBM
    pbm = initialize_pbm(train_users, config)
    
    # Fit debiaser
    debiaser = fit_debiaser(train_users, pbm, config)
    
    # Build datasets
    train_sequences, val_sequences = build_datasets(
        train_users + val_users, pbm, debiaser, config
    )
    
    # Create datasets
    train_dataset = UserDataset(
        train_sequences, 
        list(FEATURE_DIMS.keys()),
        FEATURE_DIMS,
    )
    val_dataset = UserDataset(
        val_sequences,
        list(FEATURE_DIMS.keys()),
        FEATURE_DIMS,
    )
    
    # Create model
    model_cfg = config.get("model", {})
    model = ViralFlipModel(
        feature_dims=FEATURE_DIMS,
        horizons=config.get("data", {}).get("horizons", [24, 48, 72]),
        max_lag=model_cfg.get("max_lag_bins", 12),
        l1_lambda_drift=config.get("drift_score", {}).get("l1_lambda", 0.01),
        l1_lambda_lattice=model_cfg.get("l1_lambda_w", 0.01),
        use_interactions=model_cfg.get("use_interactions", False),
        use_missing_indicators=model_cfg.get("use_missing_indicators", True),
        use_personalization=config.get("personalization", {}).get("enabled", True),
    )
    
    # Print GPU info
    if GPU_UTILS_AVAILABLE and args.device != "cpu":
        print_gpu_info()
    
    # Create trainer
    trainer = ViralFlipTrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        device=args.device,
    )
    
    # Resume from checkpoint if specified
    resume_path = Path(args.resume) if args.resume else None
    
    # Train
    history = trainer.train(train_dataset, val_dataset, resume_from=resume_path)
    
    # Save components
    save_pickle(pbm.get_state_dict(), output_dir / "pbm_state.pkl")
    save_pickle(debiaser.get_state_dict(), output_dir / "debiaser_state.pkl")
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Training complete. Best model saved to {output_dir}")
    logger.info(f"Best validation AUPRC: {max(history['val_auprc']):.4f}")


if __name__ == "__main__":
    main()

