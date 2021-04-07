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


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")

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



# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

model_exist = True
#layer = 0,1,...,42


if __name__ == "__main__":
    if model_exist==False:
        for layer_id in range(24,43):
            id_features = []
            for i, data in enumerate(trainloader, 0):
                # 准备数据
                print("layer_id=",layer_id," i=", i)
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                batch_features = net.convFeatures[layer_id]
                # print(torch.tensor(batch_features))
                id_features.extend(batch_features)
            id_features = torch.tensor(id_features)

            ocsvm = OneClassSVM(gamma='scale').fit(id_features)
            joblib.dump(ocsvm, "./model/ocsvm" + str(layer_id) + ".m")
            print("finished one class svm training for layer"+str(layer_id)+"!")

    else:
        for layer_id in range(15,26):
            ocsvm = joblib.load("./model/ocsvm" + str(layer_id) + ".m")
            # id_features = []
            # for i, data in enumerate(testloader, 0):
            #     print("id=",layer_id,", i=", i)
            #     images, labels = data
            #     images, labels = images.to(device), labels.to(device)
            #     outputs = net(images)
            #     batch_features = net.convFeatures[layer_id]
            #     id_features.extend(batch_features)
            # id_features = torch.tensor(id_features)
            #
            # preds = ocsvm.predict(id_features)
            # scores = ocsvm.score_samples(id_features)
            # id_error = 1 - np.count_nonzero(preds[:10000] == 1) / 10000
            # print(preds)
            # print(scores)
            # print("id_error=", id_error)
            #
            # with open("./result/id_scores"+str(layer_id)+".txt", "w")as f:
            #     for score in scores:
            #         f.write("%03d" % (score))
            #         f.write("\n")
            #
            # print("id数据测试完毕")

            ood_features = []
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
                    gaussian_images[j][2] = [list(map(lambda x: round(x + 0.5, 3), sublist)) for
                                             sublist in
                                             gaussian_images[j][2]]
                    gaussian_images[j] = np.clip(np.array(gaussian_images[j]), 0, 1).tolist()

                gaussian_images = torch.tensor(gaussian_images, dtype=torch.float32)
                outputs = net(gaussian_images)
                batch_features = net.convFeatures[layer_id]
                ood_features.extend(batch_features)
            ood_features = torch.tensor(ood_features)

            preds = ocsvm.predict(ood_features)
            scores = ocsvm.score_samples(ood_features)
            ood_error = 1 - np.count_nonzero(preds[:10000] == -1) / 10000
            print(preds)
            print(scores)
            print("ood_error=", ood_error)

            with open("./result/guassian_ood_scores"+str(layer_id)+".txt", "w")as f:
                for score in scores:
                    f.write("%03d" % (round(score)))
                    f.write("\n")

            print("ood 数据测试完毕")
            exit(0)



