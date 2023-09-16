import torch
import torch.nn as nn
from make_dataset_animal import animal_data_test_loader
from make_dataset_animal import visualization_test_data_loader
from Resnet import net as Resnet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from visdom import Visdom
from PIL import Image
import matplotlib.pyplot as plt
from Resnet import net
import torchvision.utils as vutils

model_path = "model_animal_sort_Resnet_29_32.pth"
#device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
device ='cpu'
model = Resnet
criteron = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_path))  # 载入模型和参数
model.eval()
writer = SummaryWriter('./new_test_logs',flush_secs=60)
sort = {0:'cat',1:'dog'}
test_loss = 0.0
correct = 0
total = 0
output = []
target = []

with torch.no_grad():  # 关闭计算图
    for batch_idx, (inputs, targets) in enumerate(animal_data_test_loader):
        sort_list = []
        sample = next(iter(animal_data_test_loader))
        outputs = model(inputs).to(device)
        loss = criteron(outputs, targets)
        test_loss += loss.item()
        _, predict = outputs.max(1)
        output.append(predict)
        target.append(targets)
        total += targets.size(0)
        correct += predict.eq(targets).sum().item()
        vutils.save_image(inputs, 'output/test_data_sample_%d.png'%batch_idx , normalize=True)

        print(batch_idx+1,'/', len(animal_data_test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        writer.add_scalar(tag="Test loss", scalar_value=test_loss, global_step=batch_idx)
