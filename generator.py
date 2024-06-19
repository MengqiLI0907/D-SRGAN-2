import torch
import torch.nn as nn

class CNNBlocks(nn.Module):
    def __init__(self, in_channels):
        super(CNNBlocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return self.conv(x) + x

class PixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels, features):
        super(Generator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1),
            nn.PReLU(),
        )
        self.RB1 = CNNBlocks(features)
        self.RB2 = CNNBlocks(features)
        self.RB3 = CNNBlocks(features)
        self.RB4 = CNNBlocks(features)

        self.mid_layer = nn.Sequential(
            nn.Conv2d(features, features * 4, 3, 1, 1),
            nn.PReLU(),
        )
        self.PS1 = PixelShuffle(features * 4, features * 4, 2)

        self.final_layer = nn.Sequential(
            nn.Conv2d(features, in_channels, 3, 1, 1),
            nn.Tanh(),
        )
# input : torch.Size([8, 1, 128, 128])
# After first_layer: torch.Size([8, 128, 128, 128])
# After RB1: torch.Size([8, 128, 128, 128])
# After RB2: torch.Size([8, 128, 128, 128])
# After RB3: torch.Size([8, 128, 128, 128])
# After RB4: torch.Size([8, 128, 128, 128])
# After mid_layer: torch.Size([8, 512, 128, 128])
# After PS1: torch.Size([8, 128, 256, 256])
# Output shape: torch.Size([8, 1, 256, 256])
    def forward(self, x):
        x1 = self.first_layer(x)
        print(f'After first_layer: {x1.shape}')
        x2 = self.RB1(x1)
        print(f'After RB1: {x2.shape}')
        x3 = self.RB2(x2)
        print(f'After RB2: {x3.shape}')
        x4 = self.RB3(x3)
        print(f'After RB3: {x4.shape}')
        x5 = self.RB4(x4)
        print(f'After RB4: {x5.shape}')
        x6 = self.mid_layer(x5 + x1)
        print(f'After mid_layer: {x6.shape}')
        x7 = self.PS1(x6)
        print(f'After PS1: {x7.shape}')
        return self.final_layer(x7)

def test():
    gen = Generator(1, 128)
    x = torch.randn(8, 1, 128, 128)
    out = gen(x)
    print(f'Output shape: {out.shape}')

if __name__ == "__main__":
    test()
