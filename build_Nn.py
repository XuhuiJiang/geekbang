import  numpy as np
import torch
from torch import nn
import random
from matplotlib import pyplot as plt
# 模型定义
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(1))
        self.bias=nn.Parameter(torch.randn(1))
    def forward(self,input):
        return (input*self.weight)+self.bias

if __name__ == '__main__':
    w=2
    b=3
    xlim=[-10,10]
    # x_train由30个-10~10的随机数组成
    x_train=np.random.randint(low=xlim[0],high=xlim[1],size=30)
    # y=2x+3+[0,1]
    y_train=[w*x+b+random.randint(0,2) for x in x_train]
    # 1、模型定义
    model=LinearModel()
    # 2、优化方法
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,weight_decay=1e-2,momentum=0.9)
    y_train=torch.tensor(y_train,dtype=torch.float32)

    for _ in range(10000):
        input=torch.from_numpy(x_train)
        output=model(input)
        # 3、得到损失函数
        loss=nn.MSELoss()(output,y_train)
        # 防止上轮的梯度与这轮累加，所以率先清零
        model.zero_grad()
        # 计算梯度（对损失函数进行求导
        loss.backward()
        optimizer.step()

    # for parameter in model.named_parameters():
    #     print(parameter)

    # 模型参数的保存
    torch.save(model.state_dict(),"./liner_model.pth")
    print(model.state_dict())

