import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5, stride=1, padding=2)
        self.r1 = nn.ReLU(inplace=True)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, 5, 1)
        self.r3 = nn.ReLU(inplace=True)
        self.s4 = nn.MaxPool2d(2, 2)

        # fcnet
        self.c5 = nn.Linear(16*54*54, 120)
        self.r5 = nn.ReLU(inplace=True)
        self.f6 = nn.Linear(120, 84)
        self.r6 = nn.ReLU(inplace=True)
        self.f7 = nn.Linear(84, 2)

    def forward(self, x):     # 3*224*224
        out = self.c1(x)      # 6*224*224
        out = self.r1(out)    # 6*224*224
        out = self.s2(out)    # 6*112*112
        out = self.c3(out)    # 16*108*108
        out = self.r3(out)    # 16*108*108
        out = self.s4(out)    # 16*54*54

        out = out.view(-1, 16*54*54)  # out.size()[0], 1*46656
        out = self.c5(out)    # 46656 --> 120
        out = self.r5(out)
        out = self.f6(out)    # 120 --> 84
        out = self.r6(out)
        out = self.f7(out)    # 84 --> 2
        return out

net = LeNet()
print(net)
# X = torch.rand(size=(1,3,128,128))
# print(net(X))