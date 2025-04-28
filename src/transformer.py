
import math, torch
import torch.nn as nn
import torch.nn.functional as F


def _pos2d(h, w, dim, dtype, device):
    half = dim // 2
    y = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)
    x = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)
    div = torch.exp(torch.arange(0, half, dtype=dtype, device=device)
                    * (-math.log(10000.0) / half)).view(1, 1, half)
    pe = torch.cat((torch.sin(y.unsqueeze(-1) * div),
                    torch.cos(y.unsqueeze(-1) * div)), -1) + \
         torch.cat((torch.sin(x.unsqueeze(-1) * div),
                    torch.cos(x.unsqueeze(-1) * div)), -1)
    return pe.view(h * w, dim)


class FusionTransformer(nn.Module):
    def __init__(self, in_chans=448, n_heads=8, n_layers=4,
                 d_ff_mult=4, out_size=(182, 218)):
        super().__init__()
        self.out_size = out_size
        self.H8 = 30
        self.W8 = 30
        enc = nn.TransformerEncoderLayer(in_chans, n_heads,
                                         in_chans * d_ff_mult,
                                         0.1, 'gelu', batch_first=False,
                                         norm_first=True)
        self.trans = nn.TransformerEncoder(enc, n_layers)
        self.up1 = self._up(in_chans, in_chans // 2)
        self.up2 = self._up(in_chans // 2, in_chans // 4)
        self.up3 = self._up(in_chans // 4, in_chans // 8)
        self.out = nn.Conv2d(in_chans // 8, 1, 1)
        self.register_buffer('pe', torch.zeros(1), persistent=False)

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if self.pe.numel() == 1:
            self.pe = _pos2d(h, w, c, x.dtype, x.device).unsqueeze(1)
        seq = x.flatten(2).permute(2, 0, 1) + self.pe
        seq = self.trans(seq)
        fused = seq.permute(1, 2, 0).reshape(b, c, h, w)
        y = self.up3(self.up2(self.up1(fused)))   # up to 240×240
        if y.shape[-2:] != self.out_size:
            y = F.interpolate(y, size=self.out_size,
                              mode='bilinear', align_corners=False)
        return self.out(y) 
