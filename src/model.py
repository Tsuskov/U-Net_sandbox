import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture as described in Ronneberger et al. (2015)"""
    
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Contracting path
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Expanding path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He's method with std = sqrt(2/N)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _crop_and_concat(self, enc_feature, dec_feature):
        """Crop encoder feature map to match decoder feature map size and concatenate"""
        # Calculate cropping amounts
        diff_h = enc_feature.size(2) - dec_feature.size(2)
        diff_w = enc_feature.size(3) - dec_feature.size(3)
        
        # Crop from all sides (symmetric)
        crop_h = diff_h // 2
        crop_w = diff_w // 2
        
        enc_cropped = enc_feature[
            :, :,
            crop_h:enc_feature.size(2) - (diff_h - crop_h),
            crop_w:enc_feature.size(3) - (diff_w - crop_w)
        ]
        
        return torch.cat([enc_cropped, dec_feature], dim=1)
    
    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Expanding path with skip connections
        x = self.upconv4(x)
        x = self._crop_and_concat(enc4, x)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = self._crop_and_concat(enc3, x)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = self._crop_and_concat(enc2, x)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = self._crop_and_concat(enc1, x)
        x = self.dec1(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x
