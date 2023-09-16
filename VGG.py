import torch
import torch.nn as nn

def vgg_block(num_convs,in_channels,out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=1))
    return nn.Sequential(*layers)

conv_arch = ((2,64),(2,128),(2,256),(2,512),(2,512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs,in_channels,out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,nn.Flatten(),
        nn.Linear(out_channels*113*113,1024),nn.ReLU(),
        nn.Dropout(0.5),nn.Linear(1024,2048),nn.ReLU(),
        nn.Dropout(0.5),nn.Linear(2048,2))

net = vgg(conv_arch)
X = torch.rand(size=(1,3,128,128))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


