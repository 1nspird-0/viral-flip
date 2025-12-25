#!/usr/bin/env python3
"""Generate synthetic multi-sensor illness episode data.

This generator creates realistic synthetic data for testing the ViralFlip
pipeline end-to-end without real patient data.

Features:
- Per-user baselines with individual variation
- Illness episodes with pre-onset physiological drift
- Behavior confounds (mobility drops, sleep changes)
- Missing data patterns
- Noise and measurement error
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from tqdm import tqdm

from viralflip.utils.io import ensure_dir, save_pickle
from viralflip.utils.seed import set_seed


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    
    n_users: int = 100
    days_per_user: int = 90
    bin_hours: int = 6
    illness_rate: float = 0.1  # Per 2-week window
    drift_magnitude: float = 2.0  # Z-score
    drift_onset_hours: int = 48
    noise_std: float = 0.3
    missing_rate: float = 0.1
    behavior_confound_strength: float = 0.5
    seed: int = 42


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
BEHAVIOR_MODALITIES = ["gps", "imu_passive", "screen"]


def generate_user_baseline(
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate individual baseline for a user.
    
    Args:
        rng: Random number generator.
        
    Returns:
        Dict mapping modality to baseline (mean, std) arrays.
    """
    baselines = {}
    
    for modality, dim in FEATURE_DIMS.items():
        # Individual variation in mean
        mean = rng.normal(0, 0.5, dim)
        # Individual variation in variability
        std = np.abs(rng.normal(1.0, 0.2, dim))
        
        baselines[modality] = {"mean": mean, "std": std}
    
    return baselines


def generate_illness_episodes(
    n_bins: int,
    illness_rate: float,
    washout_bins: int,
    rng: np.random.Generator,
) -> list[int]:
    """Generate illness onset bin indices.
    
    Args:
        n_bins: Total number of bins.
        illness_rate: Probability per 2-week window.
        washout_bins: Minimum bins between episodes.
        rng: Random number generator.
        
    Returns:
        List of onset bin indices.
    """
    onsets = []
    current_bin = washout_bins  # Start after initial washout
    
    while current_bin < n_bins:
        # Check for illness in next 2-week window
        window_bins = 14 * 4  # 14 days * 4 bins/day (6h bins)
        
        if rng.random() < illness_rate:
            # Illness occurs at random point in window
            onset = current_bin + rng.integers(1, min(window_bins, n_bins - current_bin))
            if onset < n_bins:
                onsets.append(onset)
                current_bin = onset + washout_bins
            else:
                break
        else:
            current_bin += window_bins
    
    return onsets


def generate_drift_signal(
    n_bins: int,
    onset_indices: list[int],
    drift_magnitude: float,
    drift_onset_bins: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate pre-illness drift signal.
    
    Args:
        n_bins: Total number of bins.
        onset_indices: Illness onset bin indices.
        drift_magnitude: Peak drift magnitude in z-scores.
        drift_onset_bins: Bins before onset when drift starts.
        dim: Feature dimension.
        rng: Random number generator.
        
    Returns:
        Drift signal array of shape (n_bins, dim).
    """
    drift = np.zeros((n_bins, dim))
    
    # Random feature importance (which features drift)
    feature_weights = np.abs(rng.normal(0, 1, dim))
    feature_weights = feature_weights / (feature_weights.sum() + 1e-6)
    
    for onset in onset_indices:
        # Ramp up drift before onset
        drift_start = max(0, onset - drift_onset_bins)
        
        for t in range(drift_start, onset):
            # Linear ramp
            progress = (t - drift_start) / (onset - drift_start)
            magnitude = progress * drift_magnitude
            
            # Apply to weighted features
            drift[t] += magnitude * feature_weights * rng.uniform(0.5, 1.5, dim)
    
    return drift


def generate_behavior_confound(
    n_bins: int,
    onset_indices: list[int],
    confound_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate behavior confound signal.
    
    Simulates:
    - Random mobility drops (weekends, weather, etc.)
    - Sleep pattern changes
    - Activity level changes
    
    Args:
        n_bins: Total number of bins.
        onset_indices: Illness onset indices.
        confound_strength: Strength of confounding.
        rng: Random number generator.
        
    Returns:
        Confound signal array of shape (n_bins,).
    """
    confound = np.zeros(n_bins)
    
    # Weekly pattern (weekends = lower mobility)
    for t in range(n_bins):
        day = t // 4  # 4 bins per day
        if day % 7 >= 5:  # Weekend
            confound[t] += confound_strength * rng.normal(0.5, 0.2)
    
    # Random events (travel, weather, etc.)
    n_events = int(n_bins * 0.05)
    event_starts = rng.choice(n_bins, size=n_events, replace=False)
    event_lengths = rng.integers(4, 20, size=n_events)
    
    for start, length in zip(event_starts, event_lengths):
        end = min(start + length, n_bins)
        confound[start:end] += confound_strength * rng.normal(0, 0.5)
    
    # Some confound around illness (but not causally related to drift)
    for onset in onset_indices:
        # Random behavior change around (not always before) illness
        change_start = onset + rng.integers(-8, 8)
        change_end = min(change_start + rng.integers(4, 12), n_bins)
        if 0 <= change_start < n_bins:
            confound[max(0, change_start):change_end] += confound_strength * 0.5
    
    return confound


def generate_user_data(
    user_id: str,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> dict:
    """Generate complete data for one user.
    
    Args:
        user_id: User identifier.
        config: Generation configuration.
        rng: Random number generator.
        
    Returns:
        Dict with user data.
    """
    n_bins = config.days_per_user * (24 // config.bin_hours)
    drift_onset_bins = config.drift_onset_hours // config.bin_hours
    
    # Generate baseline
    baseline = generate_user_baseline(rng)
    
    # Generate illness episodes
    washout_bins = 7 * (24 // config.bin_hours)  # 7 days
    onsets = generate_illness_episodes(
        n_bins, config.illness_rate, washout_bins, rng
    )
    
    # Generate behavior confound
    confound = generate_behavior_confound(
        n_bins, onsets, config.behavior_confound_strength, rng
    )
    
    # Generate features for each modality
    features = {}
    quality = {}
    missing = {}
    
    for modality, dim in FEATURE_DIMS.items():
        base = baseline[modality]
        
        # Base signal (individual variation around baseline)
        signal = np.zeros((n_bins, dim))
        for t in range(n_bins):
            signal[t] = base["mean"] + rng.normal(0, base["std"])
        
        # Add pre-illness drift for physiology modalities
        if modality in PHYSIOLOGY_MODALITIES:
            drift = generate_drift_signal(
                n_bins, onsets, config.drift_magnitude,
                drift_onset_bins, dim, rng
            )
            signal = signal + drift
            
            # Add confound effect (behavior affects physiology)
            confound_effect = confound[:, np.newaxis] * rng.uniform(-0.2, 0.2, dim)
            signal = signal + confound_effect
        
        # Add noise
        noise = rng.normal(0, config.noise_std, signal.shape)
        signal = signal + noise
        
        # Generate missing data
        missing_mask = rng.random(n_bins) < config.missing_rate
        
        # Quality (higher when not missing)
        q = np.ones(n_bins)
        q[missing_mask] = 0.0
        q = q * rng.uniform(0.7, 1.0, n_bins)
        
        features[modality] = signal.astype(np.float32)
        quality[modality] = q.astype(np.float32)
        missing[modality] = missing_mask
    
    # Derive labels
    labels = {}
    for horizon in [24, 48, 72]:
        horizon_bins = horizon // config.bin_hours
        lab = np.zeros(n_bins, dtype=np.int32)
        
        for onset in onsets:
            label_start = max(0, onset - horizon_bins)
            lab[label_start:onset] = 1
        
        labels[horizon] = lab
    
    # Washout mask
    washout = np.zeros(n_bins, dtype=bool)
    for onset in onsets:
        washout_end = min(onset + washout_bins, n_bins)
        washout[onset:washout_end] = True
    
    return {
        "user_id": user_id,
        "n_bins": n_bins,
        "features": features,
        "quality": quality,
        "missing": missing,
        "labels": labels,
        "onset_indices": onsets,
        "washout": washout,
        "baseline": baseline,
    }


def generate_synthetic_dataset(
    config: SyntheticConfig,
    output_dir: Path,
    n_workers: int = 1,
) -> dict:
    """Generate complete synthetic dataset.
    
    Args:
        config: Generation configuration.
        output_dir: Output directory.
        n_workers: Number of parallel workers.
        
    Returns:
        Dataset metadata dict.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    set_seed(config.seed)
    
    output_dir = ensure_dir(output_dir)
    
    def generate_single_user(args):
        """Generate data for a single user."""
        i, seed = args
        rng = np.random.default_rng(seed)
        user_id = f"user_{i:04d}"
        user_data = generate_user_data(user_id, config, rng)
        save_pickle(user_data, output_dir / f"{user_id}.pkl")
        return user_id, len(user_data["onset_indices"]), user_data["n_bins"]
    
    users = []
    total_onsets = 0
    total_bins = 0
    
    # Generate user seeds
    base_rng = np.random.default_rng(config.seed)
    user_seeds = [(i, base_rng.integers(0, 2**31)) for i in range(config.n_users)]
    
    print(f"Generating {config.n_users} users...")
    
    if n_workers > 1:
        # Parallel generation
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(generate_single_user, args) for args in user_seeds]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Users"):
                user_id, n_onsets, n_bins = future.result()
                users.append(user_id)
                total_onsets += n_onsets
                total_bins += n_bins
    else:
        # Sequential generation with progress bar
        for args in tqdm(user_seeds, desc="Users"):
            user_id, n_onsets, n_bins = generate_single_user(args)
            users.append(user_id)
            total_onsets += n_onsets
            total_bins += n_bins
    
    # Sort users for consistent ordering
    users.sort()
    
    # Save metadata
    metadata = {
        "config": config.__dict__,
        "users": users,
        "n_users": config.n_users,
        "total_onsets": total_onsets,
        "total_bins": total_bins,
        "feature_dims": FEATURE_DIMS,
        "physiology_modalities": PHYSIOLOGY_MODALITIES,
        "behavior_modalities": BEHAVIOR_MODALITIES,
    }
    
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"\nGenerated {config.n_users} users with {total_onsets} illness episodes")
    print(f"Total bins: {total_bins:,}, Illness rate: {total_onsets / config.n_users:.2f} per user")
    print(f"Estimated training samples: ~{total_bins * 0.7:,.0f}")
    print(f"Output saved to {output_dir}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic illness episode data")
    parser.add_argument("--output", "-o", type=str, default="data/synthetic",
                       help="Output directory")
    parser.add_argument("--n-users", type=int, default=100,
                       help="Number of users to generate")
    parser.add_argument("--days", type=int, default=90,
                       help="Days of data per user")
    parser.add_argument("--illness-rate", type=float, default=0.1,
                       help="Illness probability per 2-week window")
    parser.add_argument("--drift-magnitude", type=float, default=2.0,
                       help="Pre-illness drift magnitude (z-scores)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max-accuracy", action="store_true",
                       help="Generate large-scale dataset for maximum accuracy")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                       help="Number of parallel workers for generation")
    
    args = parser.parse_args()
    
    if args.max_accuracy:
        # Large-scale configuration for maximum accuracy
        print("Generating LARGE-SCALE dataset for maximum accuracy...")
        print("This will create 500 users x 180 days = 360,000 time bins")
        config = SyntheticConfig(
            n_users=500,
            days_per_user=180,
            illness_rate=0.12,
            drift_magnitude=2.5,
            drift_onset_hours=60,
            noise_std=0.25,
            missing_rate=0.08,
            behavior_confound_strength=0.6,
            seed=args.seed,
        )
    else:
        config = SyntheticConfig(
            n_users=args.n_users,
            days_per_user=args.days,
            illness_rate=args.illness_rate,
            drift_magnitude=args.drift_magnitude,
            seed=args.seed,
        )
    
    generate_synthetic_dataset(config, Path(args.output))


if __name__ == "__main__":
    main()

