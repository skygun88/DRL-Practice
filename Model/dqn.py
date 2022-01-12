import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_action) -> None:
        super().__init__()
        conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4) # (84, 84, 4) -> (20, 20, 16)
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2) # (20, 20, 16) -> (9, 9, 32)
        fc1 = nn.Linear(9*9*32, 256)
        fc2 = nn.Linear(256, n_action)

        self.conv_module = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU()
        )

        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2
        )

        if torch.cuda.is_available():
            print('cuda is available')
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()
        else:
            print('cuda is not available')

    def forward(self, x):
        out = self.conv_module(x)
        dim = 1
        for d in out.size()[1:]: #16, 4, 4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return out
