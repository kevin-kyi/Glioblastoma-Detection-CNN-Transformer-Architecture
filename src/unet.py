import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# class UNetDecoder(nn.Module):
#     def __init__(self, feature_info, decoder_channels=[256, 128, 64, 32]):
#         super().__init__()
#         # feature_info has one entry per stage; EfficientNet-B0 gives 5 → we need 4 up-stages
#         enc_chs = [f['num_chs'] for f in feature_info[::-1]]  # deepest first
#         # enc_chs = [ C32, C16, C8, C4, C2 ]
#         assert len(decoder_channels) == len(enc_chs) - 1, \
#             f"Need {len(enc_chs)-1} decoder channels, got {len(decoder_channels)}"

#         self.up_convs   = nn.ModuleList()
#         self.conv_blocks = nn.ModuleList()

#         in_ch = enc_chs[0]  # start at C32
#         for idx, out_ch in enumerate(decoder_channels):
#             skip_ch = enc_chs[idx+1]  # next shallower feature
#             # learned upsample
#             self.up_convs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
#             # conv block on [upsampled + skip]
#             self.conv_blocks.append(nn.Sequential(
#                 nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#             ))
#             in_ch = out_ch

#         # final 1×1 conv to get back to single‐channel mask
#         self.head = nn.Conv2d(in_ch, 1, kernel_size=1)

#     def forward(self, features):
#         # features: [ /2, /4, /8, /16, /32 ]
#         feats = features[::-1]  # [ /32, /16, /8, /4, /2 ]
#         x = feats[0]
#         for up, conv, skip in zip(self.up_convs, self.conv_blocks, feats[1:]):
#             x = up(x)
#             # align shapes if needed
#             if x.shape[-2:] != skip.shape[-2:]:
#                 x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
#             x = torch.cat([x, skip], dim=1)
#             x = conv(x)
#         return self.head(x)

class UNetDecoder(nn.Module):
    def __init__(self, feature_info, decoder_channels=[256,128,64,32]):
        super().__init__()
        # grab the last 5 feature sizes (coarsest last → deepest first)
        enc_chs = [f['num_chs'] for f in feature_info[::-1]]
        assert len(decoder_channels)==len(enc_chs)-1
        
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        in_ch = enc_chs[0]
        for idx, out_ch in enumerate(decoder_channels):
            skip_ch = enc_chs[idx+1]
            # 1) upsample
            self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            # 2) reduce channels to match out_ch
            self.ups.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
            # 3) conv block on [upsampled + skip]
            self.convs.append(nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # final smoothing head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, 1, kernel_size=1)
        )

    def forward(self, features):
        feats = features[::-1]
        x = feats[0]
        # every up uses two modules: upsample + channel‑reduce, then conv block
        for i in range(len(self.convs)):
            upsample, reduce_conv = self.ups[2*i], self.ups[2*i+1]
            x = upsample(x)
            # align shapes if needed
            if x.shape[-2:] != feats[i+1].shape[-2:]:
                x = F.interpolate(x, size=feats[i+1].shape[-2:], 
                                  mode='bilinear', align_corners=False)
            x = reduce_conv(x)
            x = torch.cat([x, feats[i+1]], dim=1)
            x = self.convs[i](x)
        return self.head(x)

class MultiModalSegModel(nn.Module):
    def __init__(self, encoder_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for mod in ['t1c','t1n','t2f','t2w']:
            # in_chans=1 for single‐channel input
            backbone = timm.create_model(encoder_name,
                                         pretrained=pretrained,
                                         features_only=True,
                                         in_chans=1)
            self.encoders[mod] = backbone
            self.decoders[mod] = UNetDecoder(backbone.feature_info, decoder_channels=[256,128,64,32])

    def forward(self, inputs):
        outs = {}
        for mod, x in inputs.items():
            feats = self.encoders[mod](x)
            seg  = self.decoders[mod](feats)
            outs[mod] = seg
        return outs



