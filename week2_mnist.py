import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import numpy as np
import random


# 固定所有随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
    
#数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_set = datasets.MNIST('./data',train=True,download=True,transform=transform)
test_set = datasets.MNIST('./data',train=False,download=True,transform=transform)

train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(test_set,batch_size=1000,shuffle=False)

print(f"训练集大小：{len(train_set)}")
print(f"测试集大小：{len(test_set)}")
print(f"每张图片形状：{train_set[0][0].shape}")

#定义模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Dropout(0.2),    #随机丢弃20%神经元
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64,10)
)

print(model)

#损失函数，优化器
criterion = nn.CrossEntropyLoss() #分类问题用交叉熵，不是MSE
optimizer = optim.Adam(model.parameters(),lr=0.001)


#训练函数
def train(epoch):
    model.train()
    for batch_idx,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f"Epoch={epoch} | Batch={batch_idx} | Loss={loss.item():.4f}")
            
#测试函数
def test():
    model.eval()
    correct = 0
    total = 0   
    test_loss = 0
    with torch.no_grad():
        for images,labels in test_loader:
            output = model(images)
            test_loss += criterion(output,labels).item()
            _,predicted = torch.max(output,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    print(f"\n测试集准确率：{acc:.2f}%")
    print(f"\n测试集平均Loss：{test_loss / len(test_loader):.4f}%")
    return acc


#开始训练
best_acc = 0
for epoch in range(1, 11):   # 跑10个Epoch，看它什么时候开始下降
    train(epoch)
    acc = test()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')  # 保存最好的模型
        print(f"  ↑ 新的最佳准确率！已保存")
    else:
        print(f"  ↓ 准确率下降，最佳仍是 {best_acc:.2f}%")
