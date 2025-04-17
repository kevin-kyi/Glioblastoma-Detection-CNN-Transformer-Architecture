import torch
import torch.nn as nn
import timm

class ModalityEncoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(ModalityEncoder, self).__init__()
        # Use EfficientNet-B0 from timm in features-only mode.
        # This returns a list of feature maps from various stages. We select the last one.
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        out_channels = self.backbone.feature_info[-1]['num_chs']
        # A simple segmentation head to output a single-channel prediction.
        # (For example, we simply use two conv layers with ReLU.)
        self.seg_head = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # x shape: (B, in_channels, H, W)
        features = self.backbone(x)[-1]  
        seg_out = self.seg_head(features)
        return seg_out, features