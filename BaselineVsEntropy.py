from torchvision.models.resnet import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import OneClassSVM
import joblib
import numpy as np
from numpy.random import randn
from model import ResNet
from math import log,exp
from sklearn.metrics import roc_auc_score

# 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

# 超参数设置
EPOCH = 200  # 遍历数据集次数
pre_epoch = 200  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.00001  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4, fill=0,
                          padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
#net = resnet50(pretrained=True).to(device)
#net.load_state_dict(torch.load("./model/net_200.pth"))

net = ResNet(filters_list=[16, 32, 64], N=7)
net.load_state_dict(torch.load("./model/resnet44.pth"))
# net.cuda()


# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）


def expLogits(logits):
    return [list(map(lambda x: exp(x), sublist)) for sublist in logits]

def logitsToSoftmaxs(logits):
    sm = nn.Softmax(dim=1)
    return sm(logits)

def sumExpLogits(expLogits):
    rtn = []
    for tmp in expLogits:
        rtn.append(sum(tmp))
    return rtn

def maxExpLogits(expLogits):
    rtn = []
    for tmp in expLogits:
        rtn.append(max(tmp))
    return rtn

def maxSoftmaxs(outputs):
    rtn=[]
    for tmp in outputs:
        rtn.append(torch.max(tmp).data.tolist())


def softmaxToEntropy(softmax):
    rtn=0
    for v in softmax:
        rtn -= v*log(v)
    # return max(softmax)/rtn
    return 1/rtn


def softmaxsToEntropys(outputs,sum_elogits):
    rtn = []
    for tmp in outputs:
        rtn.append(softmaxToEntropy(tmp.data.tolist()))
    if not(len(rtn)==len(sum_elogits)):
        exit(0)
    for i in range(0,len(rtn)):
        rtn[i]=sum_elogits[i]
    return rtn



if __name__ == "__main__":
    ans=[]
    for i, data in enumerate(testloader, 0):
        print("id, i=", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        # max_softmaxs = maxSoftmaxs(logitsToSoftmaxs(outputs.data.tolist()))
        # ans.extend(max_softmaxs)
        # print(max_softmaxs)

        entropys=softmaxsToEntropys(logitsToSoftmaxs(outputs),sumExpLogits(expLogits(outputs)))
        ans.extend(entropys)
        print(entropys)

    # with open("./baselineVsEntropyResult/id_baseline.txt", "w")as f:
    #     for tmp in ans:
    #         f.write("%03d" % (tmp*1000))
    #         f.write("\n")

    with open("./baselineVsEntropyResult/id_sumLogit.txt", "w")as f:
        for tmp in ans:
            f.write("%03d" % (tmp))
            f.write("\n")


    print("id数据测试完毕")

    ans = []
    # 生成10000个高斯白噪声图像
    for i in range(0, 100):
        print("ood, i=", i)
        gaussian_images = randn(100, 3, 32, 32)
        for j in range(0, len(gaussian_images)):
            gaussian_images[j][0] = [list(map(lambda x: round(x + 0.5, 3), sublist)) for
                                     sublist in
                                     gaussian_images[j][0]]
            gaussian_images[j][1] = [list(map(lambda x: round(x + 0.5, 3), sublist)) for
                                     sublist in
                                     gaussian_images[j][1]]
            gaussian_images[j][2] = [list(map(lambda x: round(x  + 0.5, 3), sublist)) for
                                     sublist in
                                     gaussian_images[j][2]]
            gaussian_images[j] = np.clip(np.array(gaussian_images[j]), 0, 1).tolist()

        gaussian_images = torch.tensor(gaussian_images, dtype=torch.float32)
        outputs = net(gaussian_images)

        # max_softmaxs = maxSoftmaxs(logitsToSoftmaxs(outputs.data.tolist()))
        # ans.extend(max_softmaxs)
        # print(max_softmaxs)

        entropys=softmaxsToEntropys(logitsToSoftmaxs(outputs),sumExpLogits(expLogits(outputs)))
        ans.extend(entropys)
        print(entropys)

    # with open("./baselineVsEntropyResult/ood_baseline.txt", "w")as f:
    #     for tmp in ans:
    #         f.write("%03d" % (tmp * 1000))
    #         f.write("\n")
    with open("./baselineVsEntropyResult/ood_sumLogit.txt", "w")as f:
        for tmp in ans:
            f.write("%03d" % (tmp))
            f.write("\n")

    print("ood 数据测试完毕")




