import torch
from torch import nn, optim
import torch.nn.functional as F


class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x)) # output: c1
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))  # output: c2[1]
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))  # output: c3[1]
        p4 = F.relu(self.p4_2(self.p4_1(x)))  # output:c4[1]
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()

        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),     #64+128+32+32=256
                                Inception(256, 64, (96, 128), (16, 32), 32),   #128+192+96+64=480
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                GlobalAvgPool2d())

        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64),
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (144, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                GlobalAvgPool2d())

        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128),
                                GlobalAvgPool2d())
        self.output = nn.Sequential(FlattenLayer(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(256, 2))

        def forward(self, x):
            x = self.b1(x)
            x = self.b2(x)
            x = self.b3(x)
            x = self.b4(x)
            x = self.b5(x)
            x = self.output(x)
            return x

net = GoogLeNet()
X = torch.rand(1, 3, 128, 128)

for blk in net.children():
    X = blk(X)
    print('output shape: ', X.shape)
