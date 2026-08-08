import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss with Class Weighting support.
    
    Formula:
        y_smooth = (1 - epsilon) * y_onehot + (epsilon / num_classes)
        Loss = - sum(y_smooth * log_softmax(logits)) * class_weight
        
    Args:
        epsilon (float): Smoothing factor in [0, 1). Default: 0.1.
        weight (torch.Tensor, optional): A manual rescaling weight given to each class.
        reduction (str): 'mean' or 'sum'.
    """
    def __init__(self, epsilon: float = 0.1, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        if not (0.0 <= epsilon < 1.0):
            raise ValueError(f"Epsilon must be in [0, 1), got {epsilon}")
        self.epsilon = epsilon
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_preds = F.log_softmax(logits, dim=-1)
        
        # Hard target NLL component
        nll_loss = -log_preds.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Uniform uncertainty component
        smooth_loss = -log_preds.mean(dim=-1)
        
        # Interpolated loss per sample
        loss = (1.0 - self.epsilon) * nll_loss + self.epsilon * smooth_loss

        # Apply class weights if provided
        if self.weight is not None:
            sample_weights = self.weight[targets]
            loss = loss * sample_weights
            if self.reduction == "mean":
                return loss.sum() / sample_weights.sum()
            elif self.reduction == "sum":
                return loss.sum()
            return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
