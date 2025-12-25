"""Main ViralFlip model combining all components.

Architecture:
1. Drift Score (φ): Compress per-modality drifts to scalar scores
2. Lag Lattice: Multi-horizon hazard model with temporal structure  
3. Interactions (optional): Sparse pairwise modality interactions
4. Personalization (optional): Per-user calibration

Also includes confidence scoring based on data quality and missingness.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from viralflip.model.drift_score import DriftScoreModule
from viralflip.model.lag_lattice import LagLatticeHazardModel
from viralflip.model.interactions import InteractionModule
from viralflip.model.personalization import PersonalizationLayer


@dataclass
class ViralFlipOutput:
    """Output container for ViralFlip predictions."""
    
    # Risk probabilities for each horizon
    risks: dict[int, float]  # horizon -> probability
    
    # Confidence scores
    confidences: dict[int, float]  # horizon -> confidence
    
    # Raw logits (before personalization)
    raw_logits: dict[int, float]
    
    # Quality summary
    quality_summary: dict[str, Any]
    
    # For interpretability
    drift_scores: dict[str, float]  # modality -> drift score
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "risks": self.risks,
            "confidences": self.confidences,
            "raw_logits": self.raw_logits,
            "quality_summary": self.quality_summary,
            "drift_scores": self.drift_scores,
        }


class ConfidenceScorer(nn.Module):
    """Score prediction confidence based on data quality."""
    
    def __init__(
        self,
        n_modalities: int,
        gamma0: float = 0.0,
        gamma1: float = 0.3,
        gamma2: float = 0.2,
        gamma3: float = 0.5,
    ):
        """Initialize confidence scorer.
        
        Args:
            n_modalities: Total number of modalities.
            gamma0: Intercept.
            gamma1: Weight for modality presence.
            gamma2: Weight for mean quality.
            gamma3: Weight for missing rate (negative effect).
        """
        super().__init__()
        
        self.n_modalities = n_modalities
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
    
    def forward(
        self,
        missing_mask: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Compute confidence scores.
        
        Args:
            missing_mask: Binary mask, shape (batch, n_modalities).
                         1 = missing, 0 = present.
            quality_scores: Quality scores, shape (batch, n_modalities).
            
        Returns:
            Confidence scores, shape (batch,).
        """
        # Number of present modalities
        n_present = (1 - missing_mask).sum(dim=-1)  # (batch,)
        
        # Mean quality of present modalities
        present_quality = quality_scores * (1 - missing_mask)
        quality_sum = present_quality.sum(dim=-1)
        mean_quality = quality_sum / (n_present + 1e-6)
        
        # Missing rate
        missing_rate = missing_mask.float().mean(dim=-1)
        
        # Confidence score
        logit = (
            self.gamma0 +
            self.gamma1 * (n_present / self.n_modalities) +
            self.gamma2 * mean_quality -
            self.gamma3 * missing_rate
        )
        
        confidence = torch.sigmoid(logit)
        
        return confidence


class ViralFlipModel(nn.Module):
    """Main ViralFlip prediction model.
    
    Combines drift scoring, lag lattice hazard model, optional interactions,
    and optional personalization.
    """
    
    # Default modalities
    MODALITIES = [
        "voice", "cough", "tap", "gait_active", 
        "rppg", "light", "baro",  # Physiology
        "gps", "imu_passive", "screen",  # Behavior (not in drift score)
    ]
    
    # Physiology modalities (for drift score)
    PHYSIOLOGY_MODALITIES = ["voice", "cough", "tap", "gait_active", "rppg", "light", "baro"]
    
    def __init__(
        self,
        feature_dims: dict[str, int],
        horizons: list[int] = [24, 48, 72],
        max_lag: int = 12,
        l1_lambda_drift: float = 0.01,
        l1_lambda_lattice: float = 0.01,
        use_interactions: bool = False,
        interaction_pairs: Optional[list[tuple[str, str]]] = None,
        l1_lambda_interaction: float = 0.1,
        use_missing_indicators: bool = True,
        use_personalization: bool = True,
        confidence_gamma: Optional[dict[str, float]] = None,
    ):
        """Initialize ViralFlip model.
        
        Args:
            feature_dims: Dict mapping modality to feature dimension.
            horizons: Prediction horizons in hours.
            max_lag: Maximum lag for lattice model (in bins).
            l1_lambda_drift: L1 regularization for drift score weights.
            l1_lambda_lattice: L1 regularization for lattice weights.
            use_interactions: Whether to use interaction module.
            interaction_pairs: Optional custom interaction pairs.
            l1_lambda_interaction: L1 regularization for interactions.
            use_missing_indicators: Whether to use missing indicators in lattice.
            use_personalization: Whether to use personalization layer.
            confidence_gamma: Optional custom confidence scoring weights.
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.horizons = horizons
        self.n_horizons = len(horizons)
        
        # Filter to physiology modalities present in feature_dims
        self.physiology_modalities = [
            m for m in self.PHYSIOLOGY_MODALITIES if m in feature_dims
        ]
        
        # Drift score module (physiology only)
        physiology_dims = {m: feature_dims[m] for m in self.physiology_modalities}
        self.drift_score = DriftScoreModule(
            modality_dims=physiology_dims,
            l1_lambda=l1_lambda_drift,
        )
        
        # Lag lattice model
        self.lag_lattice = LagLatticeHazardModel(
            n_modalities=len(self.physiology_modalities),
            horizons=horizons,
            max_lag=max_lag,
            l1_lambda=l1_lambda_lattice,
            use_missing_indicators=use_missing_indicators,
        )
        
        # Interactions (optional)
        self.use_interactions = use_interactions
        if use_interactions:
            self.interactions = InteractionModule(
                modality_names=self.physiology_modalities,
                horizons=horizons,
                interaction_pairs=interaction_pairs,
                l1_lambda=l1_lambda_interaction,
            )
        else:
            self.interactions = None
        
        # Personalization (optional)
        self.use_personalization = use_personalization
        if use_personalization:
            self.personalization = PersonalizationLayer(
                n_horizons=self.n_horizons,
            )
        else:
            self.personalization = None
        
        # Confidence scorer
        gamma = confidence_gamma or {}
        self.confidence = ConfidenceScorer(
            n_modalities=len(self.physiology_modalities),
            gamma0=gamma.get("gamma0", 0.0),
            gamma1=gamma.get("gamma1", 0.3),
            gamma2=gamma.get("gamma2", 0.2),
            gamma3=gamma.get("gamma3", 0.5),
        )
    
    def forward(
        self,
        drift_dict: dict[str, torch.Tensor],
        missing_mask: Optional[dict[str, torch.Tensor]] = None,
        quality_scores: Optional[dict[str, torch.Tensor]] = None,
        user_ids: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            drift_dict: Dict mapping modality to drift tensor.
                       Shape: (batch, seq_len, n_features).
            missing_mask: Optional dict mapping modality to missing indicator.
                         Shape: (batch, seq_len).
            quality_scores: Optional dict mapping modality to quality score.
                           Shape: (batch, seq_len) or (batch,).
            user_ids: Optional list of user IDs for personalization.
            
        Returns:
            Tuple of (risk_probs, confidence_scores).
            risk_probs: (batch, seq_len, n_horizons).
            confidence_scores: (batch, seq_len) or (batch,).
        """
        # Compute drift scores
        scores_dict = self.drift_score(drift_dict)
        
        # Build score tensor in modality order
        batch_size = next(iter(drift_dict.values())).shape[0]
        seq_len = next(iter(drift_dict.values())).shape[1]
        
        score_tensor = torch.zeros(
            batch_size, seq_len, len(self.physiology_modalities),
            device=next(iter(drift_dict.values())).device,
        )
        
        for i, modality in enumerate(self.physiology_modalities):
            if modality in scores_dict:
                score_tensor[:, :, i] = scores_dict[modality]
        
        # Build missing indicator tensor
        if missing_mask is not None:
            missing_tensor = torch.zeros_like(score_tensor)
            for i, modality in enumerate(self.physiology_modalities):
                if modality in missing_mask:
                    # Expand if needed
                    mask = missing_mask[modality]
                    if mask.dim() == 2:
                        missing_tensor[:, :, i] = mask.float()
                    else:
                        missing_tensor[:, :, i] = mask.unsqueeze(1).expand(-1, seq_len).float()
        else:
            missing_tensor = None
        
        # Lag lattice forward
        risk_probs = self.lag_lattice(score_tensor, missing_tensor)
        
        # Add interactions if enabled
        if self.use_interactions and self.interactions is not None:
            # Compute interactions per timestep
            for t in range(seq_len):
                t_scores = {m: scores_dict[m][:, t] for m in scores_dict if scores_dict[m].dim() >= 2}
                if t_scores:
                    interaction_contrib = self.interactions(t_scores)
                    risk_probs[:, t, :] = torch.sigmoid(
                        torch.logit(risk_probs[:, t, :].clamp(1e-7, 1-1e-7)) + interaction_contrib
                    )
        
        # Personalization
        if self.use_personalization and self.personalization is not None and user_ids is not None:
            risk_probs = self.personalization(risk_probs, user_ids)
        
        # Confidence scoring
        if quality_scores is not None and missing_mask is not None:
            # Build quality tensor
            quality_tensor = torch.zeros(
                batch_size, len(self.physiology_modalities),
                device=score_tensor.device,
            )
            for i, modality in enumerate(self.physiology_modalities):
                if modality in quality_scores:
                    q = quality_scores[modality]
                    if q.dim() == 2:
                        quality_tensor[:, i] = q.mean(dim=1)  # Average over seq
                    else:
                        quality_tensor[:, i] = q
            
            # Missing mask for confidence (last timestep or average)
            missing_for_conf = missing_tensor[:, -1, :] if missing_tensor is not None else torch.zeros(
                batch_size, len(self.physiology_modalities), device=score_tensor.device
            )
            
            confidence_scores = self.confidence(missing_for_conf, quality_tensor)
        else:
            confidence_scores = torch.ones(batch_size, device=score_tensor.device)
        
        return risk_probs, confidence_scores
    
    def predict_single(
        self,
        drift_dict: dict[str, np.ndarray],
        missing_mask: Optional[dict[str, bool]] = None,
        quality_scores: Optional[dict[str, float]] = None,
        user_id: Optional[str] = None,
    ) -> ViralFlipOutput:
        """Convenience method for single-sample prediction.
        
        Args:
            drift_dict: Dict mapping modality to drift array.
                       Shape: (seq_len, n_features) or (n_features,).
            missing_mask: Optional dict mapping modality to missing flag.
            quality_scores: Optional dict mapping modality to quality score.
            user_id: Optional user ID for personalization.
            
        Returns:
            ViralFlipOutput with predictions and metadata.
        """
        self.eval()
        
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            drift_tensors = {}
            for m, arr in drift_dict.items():
                t = torch.from_numpy(arr).float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)  # Add seq dim
                t = t.unsqueeze(0)  # Add batch dim
                drift_tensors[m] = t
            
            # Missing mask
            if missing_mask:
                missing_tensors = {
                    m: torch.tensor([[float(v)]]) for m, v in missing_mask.items()
                }
            else:
                missing_tensors = None
            
            # Quality
            if quality_scores:
                quality_tensors = {
                    m: torch.tensor([v]) for m, v in quality_scores.items()
                }
            else:
                quality_tensors = None
            
            # Forward
            risk_probs, confidence = self.forward(
                drift_tensors,
                missing_tensors,
                quality_tensors,
                [user_id] if user_id else None,
            )
            
            # Extract final timestep
            risk_probs = risk_probs[0, -1, :].numpy()
            confidence = confidence[0].item()
            
            # Compute drift scores for interpretability
            scores_dict = self.drift_score(drift_tensors)
            drift_scores = {
                m: scores_dict[m][0, -1].item() 
                for m in scores_dict
            }
        
        # Build output
        risks = {h: float(risk_probs[i]) for i, h in enumerate(self.horizons)}
        confidences = {h: confidence for h in self.horizons}
        raw_logits = {
            h: float(np.log(risk_probs[i] / (1 - risk_probs[i] + 1e-7)))
            for i, h in enumerate(self.horizons)
        }
        
        quality_summary = {
            "missing": missing_mask or {},
            "qualities": quality_scores or {},
        }
        
        return ViralFlipOutput(
            risks=risks,
            confidences=confidences,
            raw_logits=raw_logits,
            quality_summary=quality_summary,
            drift_scores=drift_scores,
        )
    
    def total_penalty(self) -> torch.Tensor:
        """Compute total regularization penalty."""
        penalty = self.drift_score.l1_penalty()
        penalty = penalty + self.lag_lattice.l1_penalty()
        
        if self.use_interactions and self.interactions is not None:
            penalty = penalty + self.interactions.l1_penalty()
        
        return penalty
    
    def get_state_summary(self) -> dict:
        """Get summary of learned model parameters."""
        summary = {
            "horizons": self.horizons,
            "modalities": self.physiology_modalities,
            "drift_score": self.drift_score.get_state_summary(),
            "lag_lattice": self.lag_lattice.get_state_summary(),
        }
        
        if self.use_interactions and self.interactions is not None:
            summary["interactions"] = self.interactions.get_state_summary()
        
        if self.use_personalization and self.personalization is not None:
            summary["personalization"] = self.personalization.get_state_dict()
        
        return summary

