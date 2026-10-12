\# MNIST手写数字识别 - PyTorch实现



\## 项目简介

使用PyTorch构建全连接神经网络，对MNIST手写数字数据集进行分类。



\## 环境依赖

\- Python 3.13.2

\- torch: 2.14.1+cpu

\-torchvision: 0.29.1+cpu

\-numpy: 2.5.1

\## 模型结构

\- 输入层：784维（28x28像素展平）

\- 隐藏层1：128维 + ReLU + Dropout(0.2)

\- 隐藏层2：64维 + ReLU + Dropout(0.2)

\- 输出层：10维（对应0-9十个数字）



\## 训练配置

\- 损失函数：CrossEntropyLoss

\- 优化器：Adam，学习率0.001

\- Batch Size：64

\- Epochs：10

\- 正则化：Dropout(0.2) + Early Stopping



\## 实验结果

| Epoch | 测试集准确率 |

|-------|-------------|

| 1     | 95.17%      |

| 5     | 97.62%      |

| 10    | \*\*97.83%\*\*  |



\## 运行方法

```bash

pip install torch torchvision

python week2\\\_mnist.py


