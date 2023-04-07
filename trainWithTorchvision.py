# 使用 Torchvison 中的模型进行训练我们前面说过，
# Torchvision 提供了一些封装好的网络结构，我们可以直接拿过来使用。
# 但是并没有细说如何使用它们在我们的数据集上进行训练。今天，我们就来看看如何使用这些网络结构，
# 在我们自己的数据上训练我们自己的模型。
import torch
import torchvision.models as models
from PIL import Image
import torchvision
import torchvision.transforms as transforms
if __name__ == '__main__':
    alexnet=models.alexnet(pretrained=True)
    alexnet.load_state_dict(torch.load('./model/alexnet-owt-4df8aa71.pth'))

    im=Image.open('dog.jpg')
    transform=transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    input_tensor=transform(im).unsqueeze(0)
    alexnet(input_tensor).argmax()
