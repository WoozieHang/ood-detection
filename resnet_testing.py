from torchvision.models.resnet import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from model import ResNet

# 定义是否使用GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='./model/Resnet50.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 200  # 遍历数据集次数
pre_epoch = 200  # 定义已经遍历数据集的次数
BATCH_SIZE = 512  # 批处理尺寸(batch_size)
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
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
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

# 训练
if __name__ == "__main__":
    #net2 = nn.Sequential(*list(net.children())[:-1])

    # 测试一下准确率
    print("Waiting Test!")
    correct = 0
    total = 0
    it=0
    for data in testloader:
        net.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        it = it+ 1
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))


