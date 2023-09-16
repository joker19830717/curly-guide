import os
import shutil
import cv2
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# after divide the data into different files, annotate this part
'''
data_path = "data/train/train"
train_data_path = "./dataset/train"
test_data_path = "./dataset/test"

# sort the train data and test data
for name_1 in os.listdir(data_path):
    lst = name_1.split('.')
    if lst[0]=='cat' and int(lst[1])<=8749: #0~8749 as train data
        img = cv2.imread(data_path + '/' + name_1)
        shutil.copy(data_path + '/' + name_1, "dataset/train/cat" + "/" + name_1) # copy the picture
    if lst[0]=='cat' and int(lst[1])>8749:
        img = cv2.imread(data_path + '/' + name_1)
        shutil.copy(data_path + '/' + name_1, "dataset/test/cat" + "/" + name_1)
    if lst[0]=='dog' and int(lst[1])<=8749:
        img = cv2.imread(data_path + '/' + name_1)
        shutil.copy(data_path + '/' + name_1, "dataset/train/dog" + "/" + name_1)
    if lst[0]=='dog' and int(lst[1])>8749:
        img = cv2.imread(data_path + '/' + name_1)
        shutil.copy(data_path + '/' + name_1, "dataset/test/dog" + "/" + name_1)
'''

# set the parameters
BATCH_SIZE1 = 100
BATCH_SIZE2 = 100
BATCH_SIZE3 = 1
NUM_WORKERS = 0

train_data = datasets.ImageFolder(root="./dataset/train",transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(24),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ToTensor(),
        ]))

test_data = datasets.ImageFolder(root="./dataset/test",transform = transforms.Compose(
    [
        transforms.Resize((128,128)),
        transforms.RandomCrop(96),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ToTensor(),
    ]))

visualization_test_data = datasets.ImageFolder(root="./visualization_data",transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]))

animal_data_train_loader = DataLoader(train_data,batch_size=BATCH_SIZE1,shuffle=True,num_workers = NUM_WORKERS)
animal_data_test_loader = DataLoader(test_data,batch_size=BATCH_SIZE2,shuffle = True,num_workers = NUM_WORKERS)
visualization_test_data_loader = DataLoader(visualization_test_data,batch_size=BATCH_SIZE3,shuffle=False)
