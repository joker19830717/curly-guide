from make_dataset_animal import animal_data_train_loader
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from Resnet import net as Resnet
from LeNet import LeNet
from AlexNet import my_AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')# if torch.cuda.is_available() else "cpu")
print(device)

model = Resnet
model.to(device)
model.train()

lr = 0.01
epochs = 30
criterion = nn.CrossEntropyLoss().to(device)  #使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 定义随机梯度下降优化器
# optimizer = optim.Adam(model.parameters(),
#                        lr=0.001,
#                 betas=(0.9, 0.999),
#                 eps=1e-08,
#                 weight_decay=0,
#                 amsgrad=False)
# optimizer_3 = optim.RMSprop(lr=lr,weight_decay=5e-4)
# optimizer_4 = optim.Adagrad(lr=lr,weight_decay=5e-4)
 
writer = SummaryWriter(log_dir='./logs', flush_secs=60)
#graph_inputs = torch.from_numpy(np.random.rand(1, 3, 192, 192)).type(torch.cuda.FloatTensor)
#writer.add_graph(model, (graph_inputs.to(device),))
# viz = Visdom(env="my_window")
sort = {0:'cat',1:'dog'}

for epoch in range(epochs):
    correct = 0
    total = 0
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(animal_data_train_loader):
        sort_list = []
        sample = next(iter(animal_data_train_loader))
        # viz.images(sample[0],nrow=10,win='train_data',opts=dict(title="train_data"))
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        outputs = model(inputs).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if batch_idx % 50 == 0 or batch_idx == (len(animal_data_train_loader) - 1):
        print(epoch, batch_idx, len(animal_data_train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    #writer.add_scalar("Train loss", train_loss, epoch)
    torch.save(model.state_dict(), 'model_animal_sort_Resnet_%d_32.pth' % epoch)
writer.close()

