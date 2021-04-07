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

def maxLogits(Logits):
    rtn = []
    for tmp in Logits:
        rtn.append(max(tmp))
    return rtn

def sumLogits(Logits):
    rtn = []
    for tmp in Logits:
        rtn.append(sum(tmp))
    return rtn

def expLogits(logits):
    return [list(map(lambda x: exp(x), sublist)) for sublist in logits]

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

def logitsToSoftmaxs(logits):
    sm = nn.Softmax(dim=1)
    return sm(logits)

def maxSoftmaxs(outputs):
    rtn=[]
    for tmp in outputs:
        rtn.append(torch.max(tmp).data.tolist())
    return rtn

def softmaxToEntropy(softmax):
    rtn=0
    for v in softmax:
        rtn -= v*log(v)
    return 1/rtn


def softmaxsToEntropys(softmaxs):
    rtn = []
    for tmp in softmaxs:
        rtn.append(softmaxToEntropy(tmp.data.tolist()))
    return rtn



def softmaxsToStds(softmaxs):
    rtn = []
    for tmp in softmaxs:
        rtn.append(np.std(tmp.data.tolist(),ddof=1))
    return rtn

def maxSoftmaxDivEntropy(softmaxs):
    rtn = []
    max_softmax = maxSoftmaxs(softmaxs)
    divEntropy = softmaxsToEntropys(softmaxs)
    if not (len(max_softmax) == len(divEntropy)):
        print("dim not eq!")
        exit(0)
    for i in range(0, len(max_softmax)):
        rtn[i] = max_softmax[i]*divEntropy[i]
    return rtn

def maxSoftmaxTimesStd(softmaxs):
    rtn = []
    max_softmax = maxSoftmaxs(softmaxs)
    std = softmaxsToStds(softmaxs)
    if not (len(max_softmax) == len(std)):
        print("dim not eq!")
        exit(0)
    for i in range(0, len(max_softmax)):
        rtn[i] = max_softmax[i]*std[i]
    return rtn

def stdDivEntropy(softmaxs):
    rtn = []
    std = softmaxsToStds(softmaxs)
    divEntropy = softmaxsToEntropys(softmaxs)
    if not (len(std) == len(divEntropy)):
        print("dim not eq!")
        exit(0)
    for i in range(0, len(std)):
        rtn[i] = std[i]*divEntropy[i]
    return rtn

def gaussianImages():
    gaussian_images = randn(100, 3, 32, 32)
    for j in range(0, len(gaussian_images)):
        gaussian_images[j][0] = [list(map(lambda x: round(x + 0.5, 3), sublist)) for
                                 sublist in
                                 gaussian_images[j][0]]
        gaussian_images[j][1] = [list(map(lambda x: round(x + 0.5, 3), sublist)) for
                                 sublist in
                                 gaussian_images[j][1]]
        gaussian_images[j][2] = [list(map(lambda x: round(x + 0.5, 3), sublist)) for
                                 sublist in
                                 gaussian_images[j][2]]
        gaussian_images[j] = np.clip(np.array(gaussian_images[j]), 0, 1).tolist()

    gaussian_images = torch.tensor(gaussian_images, dtype=torch.float32)
    return gaussian_images


def writeAns(file_path,ans):
    with open(file_path, "w")as f:
        for tmp in ans:
            f.write("%03d" % (tmp))
            f.write("\n")


def code0():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 0, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        max_logits=maxLogits(outputs.data.tolist())
        ans.extend(max_logits)
        print(max_logits)
    writeAns("./DetectionByLogitsResults/id_max_logit.txt",ans)
    ans = []

    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 0, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)
        max_logits = maxLogits(outputs.data.tolist())
        ans.extend(max_logits)
        print(max_logits)
    writeAns("./DetectionByLogitsResults/ood_max_logit.txt", ans)
    print("code0 finished")

def code1():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 1, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        sum_logits = sumLogits(outputs.data.tolist())
        ans.extend(sum_logits)
        print(sum_logits)
    writeAns("./DetectionByLogitsResults/id_sum_logit.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 1, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        sum_logits = sumLogits(outputs.data.tolist())
        ans.extend(sum_logits)
        print(sum_logits)
    writeAns("./DetectionByLogitsResults/ood_sum_logit.txt", ans)
    print("code1 finished")

def code2():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 2, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        sum_exp_logits = sumExpLogits(expLogits(outputs))
        ans.extend(sum_exp_logits)
        print(sum_exp_logits)
    writeAns("./DetectionByLogitsResults/id_sum_exp_logit.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 2, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        sum_exp_logits = sumExpLogits(expLogits(outputs))
        ans.extend(sum_exp_logits)
        print(sum_exp_logits)
    writeAns("./DetectionByLogitsResults/ood_sum_exp_logit.txt", ans)
    print("code2 finished")

def code3():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 3, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        max_softmaxs = maxSoftmaxs(logitsToSoftmaxs(outputs))
        ans.extend(max_softmaxs)
        print(max_softmaxs)

    writeAns("./DetectionByLogitsResults/id_max_softmax.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 3, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        max_softmaxs = maxSoftmaxs(logitsToSoftmaxs(outputs))
        ans.extend(max_softmaxs)
        print(max_softmaxs)

    writeAns("./DetectionByLogitsResults/ood_max_softmax.txt", ans)
    print("code3 finished")

def code4():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 4, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        entropys = softmaxsToEntropys(logitsToSoftmaxs(outputs))
        ans.extend(entropys)
        print(entropys)

    writeAns("./DetectionByLogitsResults/id_entropy.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 4, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        entropys = softmaxsToEntropys(logitsToSoftmaxs(outputs))
        ans.extend(entropys)
        print(entropys)

    writeAns("./DetectionByLogitsResults/ood_entropy.txt", ans)
    print("code4 finished")

def code5():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 5, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        std = softmaxsToStds(logitsToSoftmaxs(outputs))
        ans.extend(std)
        print(std)

    writeAns("./DetectionByLogitsResults/id_std.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 5, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        std = softmaxsToStds(logitsToSoftmaxs(outputs))
        ans.extend(std)
        print(std)

    writeAns("./DetectionByLogitsResults/ood_std.txt", ans)
    print("code5 finished")

def code6():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 6, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        max_softmax_div_entropy = maxSoftmaxDivEntropy(logitsToSoftmaxs(outputs))
        ans.extend(max_softmax_div_entropy)
        print(max_softmax_div_entropy)

    writeAns("./DetectionByLogitsResults/id_max_softmax_div_entropy.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 6, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        max_softmax_div_entropy = maxSoftmaxDivEntropy(logitsToSoftmaxs(outputs))
        ans.extend(max_softmax_div_entropy)
        print(max_softmax_div_entropy)

    writeAns("./DetectionByLogitsResults/ood_max_softmax_div_entropy.txt", ans)
    print("code6 finished")

def code7():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 7, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        max_softmax_times_std = maxSoftmaxTimesStd(logitsToSoftmaxs(outputs))
        ans.extend(max_softmax_times_std)
        print(max_softmax_times_std)

    writeAns("./DetectionByLogitsResults/id_max_softmax_times_std.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 7, i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        max_softmax_times_std = maxSoftmaxTimesStd(logitsToSoftmaxs(outputs))
        ans.extend(max_softmax_times_std)
        print(max_softmax_times_std)

    writeAns("./DetectionByLogitsResults/ood_max_softmax_times_std.txt", ans)
    print("code7 finished")

def code8():
    ans = []
    # 10000个cifar10图像
    for i, data in enumerate(testloader, 0):
        print("in distribution, codeId = 8, i =", i)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)

        std_div_entropy = stdDivEntropy(logitsToSoftmaxs(outputs))
        ans.extend(std_div_entropy)
        print(std_div_entropy)

    writeAns("./DetectionByLogitsResults/id_std_div_entropy.txt", ans)
    ans = []
    # 10000个高斯白噪声图像
    for i in range(0, 100):
        print("out of distribution, codeId = 8 i =", i)
        gaussian_images = gaussianImages()
        outputs = net(gaussian_images)

        std_div_entropy = stdDivEntropy(logitsToSoftmaxs(outputs))
        ans.extend(std_div_entropy)
        print(std_div_entropy)

    writeAns("./DetectionByLogitsResults/ood_std_div_entropy.txt", ans)
    print("code8 finished")

def code9():
    pass

def code10():
    pass

def code11():
    pass

def code12():
    pass

def code13():
    pass

def code14():
    pass

def code15():
    pass

def code16():
    pass

def code17():
    pass


def codeSwitch(id):
    '''
    method                                                                                               auROC
    simple indicators for absolute similarity
    0: max(logits)
    1: sum(logits)
    2: sum(exp^logits)

    simple indicators for relative similarity
    3: max(softmax)
    4: 1 / entropy of softmax
    5: std(softmax)

    high  information redundant combination between relative similarity indicators
    6: max(softmax) / entropy of softmax
    7: max(softmax) * std(softmax)
    8: std(softmax) / entropy of softmax

    low  information redundant combination between absolute and relative similarity indicators
    9: max(logits) * max-softmax
    10: max(logits) / entropy of softmax
    11: max(logits) *std(softmax)
    12: max(exp^(logits)) * max-softmax
    13: max(exp^(logits)) / entropy of softmax
    14: max(exp^(logits)) * std(softmax)
    15: sum(exp^(logits)) * max-softmax
    16: sum(exp^(logits)) / entropy of softmax
    17: sum(exp^(logits)) * std(softmax)
    '''
    if id == 0:
        code0()
    elif id == 1:
        code1()
    elif id == 2:
        code2()
    elif id == 3:
        code3()
    elif id == 4:
        code4()
    elif id == 5:
        code5()
    elif id == 6:
        code6()
    elif id == 7:
        code7()
    elif id == 8:
        code8()
    elif id == 9:
        code9()
    elif id == 10:
        code10()
    elif id == 11:
        code11()
    elif id == 12:
        code12()
    elif id == 13:
        code13()
    elif id == 14:
        code14()
    elif id == 15:
        code15()
    elif id == 16:
        code16()
    else:
        code17()


if __name__ == "__main__":
    code_id = 3
    codeSwitch(code_id)






