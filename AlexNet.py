import torch
import torch.nn as nn
import torch.nn.functional as F

class my_AlexNet(nn.Module):
    def __init__(self):
        super(my_AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48,11,4,2)
        self.pool = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(48, 128, 5, 1, 2)
        self.conv3 = nn.Conv2d(128, 192, 3, 1, 1)
        self.conv4 = nn.Conv2d(192,192, 3, 1, 1)
        self.conv5 = nn.Conv2d(192,128, 3, 1, 1)
        #self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(4608, 1000)
        self.fc4 = nn.Linear(1000,2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.num_flat_features(x))
        #x = self.drop(F.relu(self.fc1(x)))
        #x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = my_AlexNet()
#print(net)
# X = torch.rand(size=(1,3,224,224))
# print(net(X))

