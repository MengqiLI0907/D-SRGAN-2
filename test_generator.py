import torch
import torch.nn as nn
import torchvision

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
        self.PS1 = PixelShuffle(features * 4, features * 8, 2)
        self.PS2 = PixelShuffle(features * 2, features * 4, 2)

        self.final_layer = nn.Sequential(
            nn.Conv2d(features, in_channels, 3, 1, 1),
            nn.Tanh(),
        )

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
        x8 = self.PS2(x7)
        print(f'After PS2: {x8.shape}')
        return self.final_layer(x8)

def test():
    gen = Generator(1, 128)
    x = torch.randn(8, 1, 128, 128)
    out = gen(x)
    print(f'Output shape: {out.shape}')

if __name__ == "__main__":
    test()
