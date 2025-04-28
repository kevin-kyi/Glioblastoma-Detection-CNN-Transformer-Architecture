import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class UNetDecoder(nn.Module):
    def __init__(self, feature_info, decoder_channels=[256,128,64,32], output_size=(182,218)):
        super().__init__()
        self.output_size = output_size
        enc_chs = [f['num_chs'] for f in feature_info[::-1]]
        assert len(decoder_channels)==len(enc_chs)-1
        
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        in_ch = enc_chs[0]
        for idx, out_ch in enumerate(decoder_channels):
            skip_ch = enc_chs[idx+1]
            # Upsample block
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            # Skip connection processing
            self.convs.append(nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        # Final upsampling to original size
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch//2, 1, kernel_size=1)
        )

    def forward(self, features):
        feats = features[::-1]  
        x = feats[0]
        
        for i, (up, conv) in enumerate(zip(self.ups, self.convs)):
            x = up(x)
            # Handle potential size mismatches
            if x.shape[-2:] != feats[i+1].shape[-2:]:
                x = F.interpolate(x, size=feats[i+1].shape[-2:], 
                                mode='bilinear', align_corners=False)
            x = torch.cat([x, feats[i+1]], dim=1)
            x = conv(x)
            
        return self.head(x)


class MultiModalSegModel(nn.Module):
    def __init__(self, encoder_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for mod in ['t1c','t1n','t2f','t2w']:
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



