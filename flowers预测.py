import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import transforms
import os
from torch.utils.tensorboard import SummaryWriter

torch.random.manual_seed(0)
writer = SummaryWriter()
device = torch.device('cuda')


# 将经典网络修改成自己需要的
def resnet18_model(weight_path):
    model = resnet18(pretrained=False)
    in_features = model.fc.in_features
    # 十四分类
    model.fc = nn.Linear(in_features=in_features, out_features=14)
    weight_dict = torch.load(weight_path)
    weight_dict.pop('fc.weight')
    weight_dict.pop('fc.bias')
    my_resnet18_dict = model.state_dict()
    my_resnet18_dict.update(weight_dict)
    model.load_state_dict(my_resnet18_dict)
    return model


def buid_data(image_train_path, image_val_path):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转 ±10 度
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # 随机调整亮度、对比度、饱和度、色调
        transforms.ToTensor(),  # 转换为 Tensor
    ])

    train_data = ImageFolder(root=image_train_path, transform=transform)
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_data = ImageFolder(root=image_val_path, transform=transform)
    val_data_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    classes_names = val_data.classes
    return train_data_loader, train_data, val_data_loader, val_data, classes_names


def train_passer(model, epochs, lr, train_data_loader, train_data, model_path):
    best_acc = 0
    net = model
    # 继续训练
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 添加学习率衰减
    cre = nn.CrossEntropyLoss()
    # 训练
    for epoch in range(epochs):
        # 每轮预测正确的个数
        acc_total = 0
        # 每轮损失
        loss_total = 0
        for batch_idx, (x, y) in enumerate(train_data_loader):
            x = x.to(device)
            y = y.to(device)
            pre = net(x)
            loss = cre(pre, y)
            loss_total += loss.item()
            acc_total += sum(torch.argmax(pre, dim=1) == y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 100 == 0:
                grid = torchvision.utils.make_grid(x)
                writer.add_image(f'my_image_batch/{batch_idx}', grid, 0)
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        scheduler.step()
        print(f'lr={current_lr}')  # 打印当前学习率
        print(f'epoch: {epoch}, loss: {loss_total / len(train_data)}, acc: {acc_total / len(train_data)}')
        acc = acc_total / len(train_data)
        # 仅保存验证集性能最佳的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), model_path)
        writer.add_scalar("Loss/train", loss_total / len(train_data), epoch)
        writer.add_scalar("acc/train", acc_total / len(train_data), epoch)
        writer.flush()


def val_passer(model, val_data_loader, val_data, model_path, classes_names, csv_path):
    # 使用pd创建一个csv，总共16列
    total_data = np.empty(shape=(0, 16))
    model.eval()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict())
    with torch.no_grad():
        acc_total = 0
        for batch_idx, (x, y) in enumerate(val_data_loader):
            x = x.to(device)
            y = y.to(device)
            pre = model(x)
            # 将pre装成numpy
            p1 = pre.detach().cpu().numpy()
            pre1 = torch.argmax(pre, dim=1)
            # (64,)
            p2 = pre1.unsqueeze(dim=1).detach().cpu().numpy()
            y1 = y.unsqueeze(dim=1).detach().cpu().numpy()
            # 每个批次将预测的结果数据和真实标签拼接成64行16列
            batch_data = np.concatenate((p1, p2, y1), axis=1)
            total_data = np.concatenate((total_data, batch_data), axis=0)
            acc_total += torch.sum(pre1 == y)
        print(f"acc_total:{acc_total / len(val_data)}")
        cols = [*classes_names, "predict", "label"]
        pd.DataFrame(data=total_data, columns=cols).to_csv(csv_path, index=False)

        print(f'val_acc: {acc_total / len(val_data)}')


if __name__ == '__main__':
    image_train_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'flowers', 'train'))
    image_val_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'flowers', 'val'))
    model_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'weights', 'my_flowers_resnet18.pth'))
    weight_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'weights', 'resnet18.pth'))
    csv_path = os.path.relpath(os.path.join(os.path.dirname(__file__), 'result', 'result1.csv'))
    model = resnet18_model(weight_path)
    epochs = 40
    lr = 0.001
    train_data_loader, train_data, val_data_loader, val_data, classes_names = buid_data(image_train_path,
                                                                                        image_val_path)
    train_passer(model, epochs, lr, train_data_loader, train_data, model_path)
    val_passer(model, val_data_loader, val_data, model_path, classes_names, csv_path)
