"""Loss functions for multi-horizon illness prediction.

Key considerations:
- Imbalanced classes (rare positive events)
- Multiple horizons with different weights
- Focal loss to focus on hard examples
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Focuses learning on hard examples by down-weighting easy negatives.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ):
        """Initialize focal loss.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples).
            alpha: Class weight for positive class. If None, no weighting.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Predicted probabilities, shape (batch, ...).
            targets: Binary targets, shape (batch, ...).
            weights: Optional sample weights, shape (batch, ...).
            
        Returns:
            Loss value.
        """
        # Clamp for numerical stability
        p = inputs.clamp(1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        bce = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        
        # Focal weight
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_t
        
        # Apply focal weighting
        loss = focal_weight * bce
        
        # Apply sample weights
        if weights is not None:
            loss = loss * weights
        
        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiHorizonLoss(nn.Module):
    """Multi-horizon loss combining losses across prediction horizons.
    
    L = Σ_H w_H * L_H(y_H, r_H)
    """
    
    def __init__(
        self,
        horizons: list[int],
        horizon_weights: Optional[list[float]] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        pos_weight: float = 5.0,
    ):
        """Initialize multi-horizon loss.
        
        Args:
            horizons: List of prediction horizons.
            horizon_weights: Weights for each horizon. If None, equal weights.
            use_focal: Whether to use focal loss.
            focal_gamma: Focal loss gamma parameter.
            pos_weight: Weight for positive class.
        """
        super().__init__()
        
        self.horizons = horizons
        self.n_horizons = len(horizons)
        
        if horizon_weights is None:
            horizon_weights = [1.0] * self.n_horizons
        self.horizon_weights = torch.tensor(horizon_weights)
        
        self.use_focal = use_focal
        self.pos_weight = pos_weight
        
        if use_focal:
            # Alpha is derived from pos_weight
            alpha = pos_weight / (1 + pos_weight)
            self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=alpha, reduction="none")
        else:
            self.loss_fn = None
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-horizon loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch, [seq,] n_horizons).
            targets: Binary targets, shape (batch, [seq,] n_horizons).
            mask: Optional mask for valid samples, shape (batch, [seq]).
            
        Returns:
            Tuple of (total_loss, loss_dict with per-horizon losses).
        """
        device = predictions.device
        self.horizon_weights = self.horizon_weights.to(device)
        
        # Handle sequence dimension
        if predictions.dim() == 3:
            # (batch, seq, n_horizons) -> flatten batch and seq
            batch, seq, n_h = predictions.shape
            predictions = predictions.view(-1, n_h)
            targets = targets.view(-1, n_h)
            if mask is not None:
                mask = mask.view(-1)
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        for i, horizon in enumerate(self.horizons):
            pred_h = predictions[:, i]
            target_h = targets[:, i]
            
            if self.use_focal:
                loss_h = self.loss_fn(pred_h, target_h)
            else:
                # Weighted BCE
                pos_weight = torch.tensor([self.pos_weight], device=device)
                loss_h = F.binary_cross_entropy(
                    pred_h, target_h, reduction="none"
                )
                # Apply pos_weight manually
                weight = target_h * self.pos_weight + (1 - target_h)
                loss_h = loss_h * weight
            
            # Apply mask
            if mask is not None:
                loss_h = loss_h * mask.float()
                loss_h = loss_h.sum() / (mask.sum() + 1e-6)
            else:
                loss_h = loss_h.mean()
            
            loss_dict[f"loss_{horizon}h"] = loss_h
            total_loss = total_loss + self.horizon_weights[i] * loss_h
        
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """Combined loss with prediction loss and regularization."""
    
    def __init__(
        self,
        horizons: list[int],
        horizon_weights: Optional[list[float]] = None,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
        pos_weight: float = 5.0,
        reg_weight: float = 1.0,
    ):
        """Initialize combined loss.
        
        Args:
            horizons: List of prediction horizons.
            horizon_weights: Weights for each horizon.
            use_focal: Whether to use focal loss.
            focal_gamma: Focal loss gamma parameter.
            pos_weight: Weight for positive class.
            reg_weight: Weight for regularization penalty.
        """
        super().__init__()
        
        self.prediction_loss = MultiHorizonLoss(
            horizons=horizons,
            horizon_weights=horizon_weights,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
        )
        self.reg_weight = reg_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reg_penalty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.
        
        Args:
            predictions: Predicted probabilities.
            targets: Binary targets.
            reg_penalty: Regularization penalty from model.
            mask: Optional valid sample mask.
            
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        pred_loss, loss_dict = self.prediction_loss(predictions, targets, mask)
        
        reg_loss = self.reg_weight * reg_penalty
        total_loss = pred_loss + reg_loss
        
        loss_dict["reg"] = reg_loss
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict

