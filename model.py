import torch.nn as nn
import torch
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_poll_stride):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_poll_stride = max_poll_stride

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=self.max_poll_stride)
         )

    def forward(self, x):
        return self.conv_block(x)


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv_layer = nn.Sequential(
            ConvBlock(3, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 128, 2)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=128*8*8, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=8)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0),-1)
        x = self.linear_layer(x)
        return x


# Checking the model
if __name__ == '__main__':
    model = EmotionClassifier().to('cuda:0')
    summary(model, (3, 90, 90))
