import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class DNN(nn.Module):
    def __init__(self, inputSize, width, deep, outputSize, activateFunc=nn.ReLU(), **kwargs):
        super(DNN,self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        begLinear = nn.Linear(inputSize, width)
        middleLinear = nn.Linear(width, width)
        endLinear = nn.Linear(width, outputSize)
        self.modelList = [begLinear, activateFunc]
        for i in range(deep):
            self.modelList.append(middleLinear)
            self.modelList.append(activateFunc)
        self.modelList.append(endLinear)
        self.layers=nn.ModuleList(self.modelList)

    def forward(self, x):
        x = x.view(-1,self.inputSize)
        for layer in self.layers:
            x = layer(x)
        return x

    def getOutputSize(self):
        return self.outputSize

class normalizeDNN(DNN):
    def __init__(self, inputSize, width, deep, outputSize, activateFunc=nn.ReLU(), tau = 1/2, **kwargs):
        super(normalizeDNN,self).__init__(inputSize, width, deep, outputSize, activateFunc=nn.ReLU())
        self.tau = tau

    def forward(self, x):
        x = x.view(-1,self.inputSize)
        for layer in self.modelList:
            if isinstance(layer, nn.ReLU):
                x = layer(x)
            else:
                weight = nn.functional.normalize(layer.weight, dim = 0)
                x = self.tau * x.mm(weight.T)
        return x

class ResNet18(nn.Module):
    def __init__(self, outputSize, **kwargs):
        super(ResNet18,self).__init__()
        self.outputSize = outputSize
        self.resnet = resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features  
        self.resnet.fc = nn.Linear(num_ftrs, self.outputSize)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def getOutputSize(self):
        return self.outputSize


modelDict = {'DNN':DNN, 'ResNet18':ResNet18, 'normalizeDNN':normalizeDNN}

def getModel(modelName='DNN'):
    return modelDict[modelName]
