import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionTransformer(nn.Module):
    """
    Fuse modality‑specific CNN feature maps with a Vision‑Transformer encoder.

    Args
    ----
    feature_dim : int
        Number of channels after concatenating all modality encoders
        (your loop passes 1280).
    num_layers  : int
        How many TransformerEncoder layers.
    num_heads   : int
        Multi‑head attention heads.
    num_classes : int
        Output channels (1 for binary mask).
    """
    def __init__(self,
                 feature_dim: int = 1280,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 num_classes: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        # --- Positional encoding (fixed, 2‑D sine/cos) ----------------------
        self.register_buffer("pe", torch.zeros(1), persistent=False)  # lazy‑init

        # --- Transformer encoder -------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,   # we will feed (S, B, C)
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # --- 1×1 conv segmentation head ------------------------------------
        self.seg_head = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    # --------------------------------------------------------------------- #
    # helper: build 2‑D sin‑cos positional encoding (no learnable weights)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_2d_sincos_pos_embed(h, w, dim, dtype, device):
        """
        Return a (h*w, dim) fixed 2‑D sine/cos positional encoding
        """
        assert dim % 2 == 0, "feature_dim must be even"
        half_dim = dim // 2

        # coordinate vectors
        y = torch.arange(h, dtype=dtype, device=device).unsqueeze(1)      # (H,1)
        x = torch.arange(w, dtype=dtype, device=device).unsqueeze(0)      # (1,W)

        # frequency coefficients
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=dtype, device=device)
            * (-math.log(10000.0) / half_dim)
        )                                                                 # (half_dim,)

        # *** add final singleton dim to broadcast correctly ***
        y = y.unsqueeze(-1)                       # (H,1,1)
        x = x.unsqueeze(-1)                       # (1,W,1)
        div_term = div_term.unsqueeze(0).unsqueeze(0)   # (1,1,half_dim)

        # (H,W,half_dim)
        pe_y = torch.sin(y * div_term), torch.cos(y * div_term)
        pe_x = torch.sin(x * div_term), torch.cos(x * div_term)

        pe_y = torch.cat(pe_y, dim=-1)            # (H,W,dim)
        pe_x = torch.cat(pe_x, dim=-1)            # (H,W,dim)

        pe = pe_y + pe_x                          # (H,W,dim)
        return pe.view(h * w, dim)                # (H*W, dim)

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)  – concatenated CNN feature maps
        returns logits : (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        # flatten spatial→sequence: (H*W, B, C)
        seq = x.flatten(2).permute(2, 0, 1)

        # build / cache positional enc. once we know H*W
        if self.pe.numel() == 1 or self.pe.size(0) != H * W:
            self.pe = self._build_2d_sincos_pos_embed(
                H, W, C, dtype=seq.dtype, device=seq.device
            ).unsqueeze(1)  # (S,1,C)

        seq = seq + self.pe                       # add position information
        seq = self.transformer(seq)               # (S, B, C)

        # reshape back to (B,C,H,W)
        fused = seq.permute(1, 2, 0).reshape(B, C, H, W)

        # 1×1 conv head → logits
        logits = self.seg_head(fused)
        return logits