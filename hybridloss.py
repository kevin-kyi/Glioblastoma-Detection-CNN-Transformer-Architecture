import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- #
# Dice loss (works for binary masks; extend for multi‑class if needed)
# --------------------------------------------------------------------- #
def dice_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """
    logits  : (B, 1, H, W) ‑ or (B, C, H, W) for multi‑class w/ one‑hot targets
    targets : same shape as logits after converting to float
    """
    probs   = torch.sigmoid(logits)
    targets = targets.float()

    # Sum over spatial dims, keep batch / channel dims
    dims = tuple(range(2, logits.ndim))   # (2,3) for (B,C,H,W)
    intersection = (probs * targets).sum(dims)
    union        = probs.sum(dims) + targets.sum(dims)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()              # 1‑Dice  → loss


# --------------------------------------------------------------------- #
# Focal loss (binary version)
# --------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """
    Binary Focal loss (α‑balanced, γ‑modulated).
    """
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t: probability of the true class
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_term = (self.alpha * targets + (1 - self.alpha) * (1 - targets))
        focal_term = focal_term * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "sum":
            return focal_term.sum()
        elif self.reduction == "none":
            return focal_term
        else:                               # "mean" (default)
            return focal_term.mean()


# --------------------------------------------------------------------- #
# Hybrid Dice + Focal loss
# --------------------------------------------------------------------- #
class HybridDiceFocalLoss(nn.Module):
    """
    Combined loss = w_dice * DiceLoss + w_focal * FocalLoss
    """
    def __init__(self,
                 dice_weight:  float = 0.7,
                 focal_weight: float = 0.3,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 smooth: float = 1e-6):
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.smooth       = smooth
        self.focal_loss   = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d_loss = dice_loss(logits, targets, smooth=self.smooth)
        f_loss = self.focal_loss(logits, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss