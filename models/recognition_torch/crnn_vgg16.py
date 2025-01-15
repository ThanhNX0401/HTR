import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn

class Network(nn.Module):
    """ CRNN network with VGG16_BN backbone for CTC loss"""
    def __init__(self, num_chars: int, dropout: float=0.2, pretrained: bool=True):
        super(Network, self).__init__()
        
        # Load pretrained VGG16_BN and get feature extractor
        model = vgg16_bn(pretrained=None)
        
        # # Remove last max pooling and all classifier layers
        # features = list(vgg.features)
        # # Modify the last two MaxPool2d layers to have stride=(2,1)
        # features[23] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 4th max pooling
        # features[33] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 5th max pooling
        
        # self.features = nn.Sequential(*features)
        
        pool_idcs = [idx for idx, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
        # Replace their kernel with rectangular ones
        for idx in pool_idcs[-3:]:
            model.features[idx] = nn.MaxPool2d((2, 1))
        # Patch average pool & classification head
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(512, num_chars)
        
        self.feat_extractor = model.features

        with torch.inference_mode():
            out_shape = self.feat_extractor(torch.zeros((1, 3, 32, 128))).shape
        lstm_in = out_shape[1] * out_shape[2]
        
        # Sequence processing
        self.lstm1 = nn.LSTM(lstm_in, 256, bidirectional=True, num_layers=2, batch_first=True)
        # self.lstm_dropout1 = nn.Dropout(p=dropout)
        
        # Final classifier
        self.output = nn.Linear(512, num_chars + 1)  # +1 for CTC blank

        # for n, m in self.named_modules():
        #     # Don't override the initialization of the backbone
        #     if n.startswith("feat_extractor."):
        #         continue
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1.0)
        #         m.bias.data.zero_()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # normalize images between 0 and 1
        print(images.shape)
        x = images / 255.0
        x=x.permute(0, 3, 1, 2)
        features = self.feat_extractor(x)
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        
        # # transpose image to channel first
        # x = x.permute(0, 3, 1, 2)
        
        # # CNN features
        # x = self.features(x)
        
        # # Prepare for sequence processing
        # # (batch, channels, height, width) -> (batch, width, channels * height)
        # b, c, h, w = x.size()
        # x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        # x = x.reshape(b, w, c * h)
        
        # RNN layers
        x, _ = self.lstm1(features_seq)
        # x = self.lstm_dropout1(x)
        
        # Final classifier
        x = self.output(x)
        
        x = F.log_softmax(x, 2)
        
        # x = x.permute(1, 0, 2)
        # probs = F.log_softmax(x, dim=-1)
        
        return x