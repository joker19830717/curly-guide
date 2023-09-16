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
import os
import cv2
torch.onnx
from AlexNet import my_AlexNet

# change the .pth model into .onnx model
# model_test = my_AlexNet()
# model_statedict = torch.load("model_animal_sort_AlexNet_49.pth",map_location=lambda storage,loc:storage)   #导入Gpu训练模型，导入为cpu格式
# print(model_statedict)
# model_test.load_state_dict(state_dict=model_statedict)  #将参数放入model_test中
# model_test.eval()  # 测试，看是否报错
# #下面开始转模型，cpu格式下
# device = torch.device("cpu")
# dummy_input = torch.randn(1, 3, 224, 224,device=device)
# input_names = ["input"]
# output_names = ["output"]
# torch.onnx.export(model_test, dummy_input, "model_Alexnet_animal.onnx", opset_version=9, verbose=False, output_names=["hm"])

model_path = "model_Resnet34_128_50_on_train.pth"
#device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = Resnet
criteron = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_path))  # 载入模型和参数
model.eval()
sort = {0:'cat',1:'dog'}
test_loss = 0.0
correct = 0
total = 0
output = []
target = []

sort_list = []
with torch.no_grad():  # 关闭计算图
    for batch_idx, (inputs, targets) in enumerate(visualization_test_data_loader):
        sample = next(iter(animal_data_test_loader))
        outputs = model(inputs).to(device)
        loss = criteron(outputs, targets)
        test_loss += loss.item()
        _, predict = outputs.max(1)
        output.append(predict)
        target.append(targets)
        total += targets.size(0)
        correct += predict.eq(targets).sum().item()
        for i in range(len(targets)):
            sort_result = sort[list(targets.numpy())[i]] + "/" + sort[list(predict.numpy())[i]]
            sort_list.append(sort_result)
        # -------------------------------------------------------------------
        print(batch_idx+1,'/', len(animal_data_test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

error = []
error_num = 0

for i in range(100):
    if sort_list[i][:3] != sort_list[i][4:]:
        #print(sort_list[i][:3],sort_list[i][3:])
        error_num += 1
        error.append(i)
print(error)

path = "picture"
plt.figure()
plt.subplot(3,6,1)
error =[12, 17, 18, 24, 27, 29, 30, 33, 40, 57, 62, 63, 65, 83, 87, 92]
right = []
lst = os.listdir(path)
fig,axes=plt.subplots(nrows=5,ncols=3,figsize=(10,10))  #设定n*n排列方式，这里设置的是2*2，nrows行，ncols列，figsize设定窗口大小
for i in range(15):
    img = plt.imread(path+'/'+lst[right[i]])
    print(i//3,i%3)
    axes[i//3,i%3].imshow(img)

plt.show()

# size = [32,64,96,128,160]
# acc = [0.7956,0.83587,0.8864,0.9020,0.9180]
#
# plt.subplot()
# plt.plot(size,acc)
# plt.title('acc change with img size')
# plt.show()