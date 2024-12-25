import torch
import torch.nn as nn
import torch.nn.functional as F

def activation_layer(activation: str="relu", alpha: float=0.1, inplace: bool=True):
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)

    Returns:
        torch.Tensor: activation layer
    """
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)


class ConvBlock(nn.Module):
    """ Convolutional block with batch normalization
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conv=True, stride=1, dropout=0.2, activation="leaky_relu"):
        super(ResidualBlock, self).__init__()
        self.convb1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act1 = activation_layer(activation)

        self.convb2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=dropout)
        
        self.shortcut = None
        if skip_conv:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act2 = activation_layer(activation)
        
    def forward(self, x):
        skip = x
        
        out = self.act1(self.convb1(x))
        out = self.convb2(out)

        if self.shortcut is not None:
            out += self.shortcut(skip)

        out = self.act2(out)
        out = self.dropout(out)
        
        return out

class Network(nn.Module):
    """ Handwriting recognition network based on ResNet18 architecture for CTC loss"""
    def __init__(self, num_chars: int, activation: str="leaky_relu", dropout: float=0.2):
        super(Network, self).__init__()
        
        # Initial convolution layer
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.act1 = activation_layer(activation)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rb1 = ResidualBlock(3, 64, skip_conv = True, stride=1, activation=activation, dropout=dropout)
        # ResNet18 layers
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, skip_conv=False, stride=1, dropout=dropout, activation=activation),
            ResidualBlock(64, 64, skip_conv=False, stride=1, dropout=dropout, activation=activation)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, skip_conv=True, stride=2, dropout=dropout, activation=activation),
            ResidualBlock(128, 128, skip_conv=False, stride=1, dropout=dropout, activation=activation)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, skip_conv=True, stride=2, dropout=dropout, activation=activation),
            ResidualBlock(256, 256, skip_conv=False, stride=1, dropout=dropout, activation=activation)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, skip_conv=True, stride=2, dropout=dropout, activation=activation),
            ResidualBlock(512, 512, skip_conv=False, stride=1, dropout=dropout, activation=activation)
        )
        
        # LSTM layers remain the same
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout1 = nn.Dropout(p=dropout)
        
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout2 = nn.Dropout(p=dropout)
        
        self.output = nn.Linear(512, num_chars + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # normalize images between 0 and 1
        x = images / 255.0
        
        # transpose image to channel first
        x = x.permute(0, 3, 1, 2)
        
        # Initial convolution
        x = self.rb1(x)
        # x = self.bn1(x)
        # x = self.act1(x)
        # x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Reshape and LSTM layers
        x = x.reshape(x.size(0), -1, x.size(1))
        
        x, _ = self.lstm1(x)
        x = self.lstm_dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.lstm_dropout2(x)
        
        x = self.output(x)
        x = F.log_softmax(x, 2)
        
        return x