import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

class Network(nn.Module):
    """ CRNN network with VGG16_BN backbone for CTC loss"""
    def __init__(self, num_chars: int, dropout: float=0.2, pretrained: bool=True):
        super(Network, self).__init__()
        
        # Load pretrained VGG16_BN and get feature extractor
        vgg = vgg16_bn(pretrained=pretrained)
        
        # Remove last max pooling and all classifier layers
        features = list(vgg.features)
        # Modify the last two MaxPool2d layers to have stride=(2,1)
        features[23] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 4th max pooling
        features[33] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 5th max pooling
        
        self.features = nn.Sequential(*features)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        # Sequence processing
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.lstm_dropout1 = nn.Dropout(p=dropout)
        
        # Final classifier
        self.output = nn.Linear(512, num_chars + 1)  # +1 for CTC blank

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # normalize images between 0 and 1
        x = images / 255.0
        
        # transpose image to channel first
        x = x.permute(0, 3, 1, 2)
        
        # CNN features
        x = self.features(x)
        
        # x = self.avgpool(x)
        # # Prepare for sequence processing
        # # (batch, channels, 1, width) -> (batch, width, channels)
        # b, c, h, w = x.size()
        # x = x.squeeze(2).permute(0, 2, 1)
        
        # Prepare for sequence processing
        # (batch, channels, height, width) -> (batch, width, channels * height)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.reshape(b, w, c * h)
        
        # RNN layers
        x, _ = self.lstm1(x)
        x = self.lstm_dropout1(x)
        
        # Final classifier
        x = self.output(x)
        x = F.log_softmax(x, 2)
        
        return x