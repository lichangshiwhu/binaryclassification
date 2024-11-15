import torch
import torch.nn as nn
import torch.nn.functional as F

import globalVar

class LeastSquareLoss(nn.Module):
    def forward(self, output, target):
        outputLen = len(output)
        subLables = output - target
        return torch.dot(subLables, subLables) / outputLen

class HingeLoss(nn.Module):
    def forward(self, output, target):
        outputLen = len(output)
        hingeloss = 1 - torch.mul(output, target)
        hingeloss[hingeloss < 0] = 0
        res = torch.sum(hingeloss) / outputLen
        return res

class SquaredHingeLoss(nn.Module):
    def forward(self, output, target):
        outputLen = len(output)
        hingeloss = 1 - torch.mul(output, target)
        hingeloss[hingeloss < 0] = 0
        return torch.dot(hingeloss, hingeloss) / outputLen

class LogisticLoss(nn.Module):
    def forward(self, output, target):
        outputLen = len(output)
        logisitcloss = torch.log(torch.exp(-torch.mul(output, target)) + 1)
        res = torch.sum(logisitcloss) / outputLen
        return res

class SigmoidLoss(nn.Module):
    def __init__(self, lossScale = 1):
        super(SigmoidLoss, self).__init__()
        self.lossScale = lossScale

    def forward(self, output, target):
        outputLen = len(output)
        sigmoidloss = 2 * torch.sigmoid(-2 * self.lossScale * torch.mul(output, target))
        res = torch.sum(sigmoidloss) / outputLen
        return res

class RampLoss(nn.Module):
    def __init__(self, lossScale = -1):
        super(RampLoss, self).__init__()
        self.lossScale = lossScale

    def forward(self, output, target):
        outputLen = len(output)
        Yfx = torch.mul(output, target)
        ramploss = F.relu(1 - Yfx) - F.relu(self.lossScale - Yfx)
        res = torch.sum(ramploss) / outputLen
        return res

class WelschLoss(nn.Module):
    def __init__(self, lossScale = 2):
        super(WelschLoss, self).__init__()
        self.lossScale = lossScale

    def forward(self, output, target):
        outputLen = len(output)
        welschloss = output - target
        welschloss = -welschloss * welschloss / (2 * self.lossScale * self.lossScale)
        welschloss = (1 - torch.exp(welschloss))*self.lossScale*self.lossScale/2
        res = torch.sum(welschloss) / outputLen
        return res

class SavageLoss(nn.Module):
    def forward(self, output, target):
        outputLen = len(output)
        sigmoidloss = torch.sigmoid(-2 * torch.mul(output, target))
        sigmoidloss = sigmoidloss * sigmoidloss
        res = torch.sum(sigmoidloss) / outputLen
        return res

class FocalLoss(nn.Module):
    def __init__(self, lossScale = 1, **kwargs):
        super(FocalLoss, self).__init__()
        self.lossScale = lossScale

    def forward(self, output, target):
        outputLen = len(output)
        t = torch.mul(output, target)
        logisitcloss = torch.log(torch.exp(-t) + 1)
        if self.lossScale >= 0:
            weights = torch.sigmoid(-t) ** self.lossScale
        else:
            # smooth
            weights = (torch.sigmoid(-t) + 1) ** self.lossScale
        focalloss = weights * logisitcloss
        res = torch.sum(focalloss) / outputLen
        return res

class MAILLoss(nn.Module):
    def __init__(self, lossScale1 = 1, lossScale2 = 1, **kwargs):
        super(MAILLoss, self).__init__()
        self.lossScale1 = lossScale1
        self.lossScale2 = lossScale2

    def forward(self, output, target):
        outputLen = len(output)
        t = torch.mul(output, target)
        prob = torch.sigmoid(t)

        pm = 2 * prob - 1
        weights = torch.sigmoid(-self.lossScale1*(pm - self.lossScale2))
        sum_weights = torch.sum(weights)
        weights = weights / sum_weights
        # logisitc loss is the cross entrpy loss in binary classification
        mailloss = weights * torch.log(torch.exp(-t) + 1)
        res = torch.sum(mailloss) / outputLen
        return res

class SOVRLoss(nn.Module):
    def __init__(self, lossScale1 = 1, lossScale2 = 50, **kwargs):
        super(SOVRLoss, self).__init__()
        # hyperparrameter to balance the loss
        self.lossScale1 = lossScale1
        # top M% data points in minibatch for uncertain classfication
        # => the least M% data ponits in minibatch of probability  
        self.lossScale2 = lossScale2

    def forward(self, output, target):
        outputLen = len(output)
        t = torch.mul(output, target)
        k = int(outputLen * self.lossScale2 * 0.01)
        sum_loss2 = 0
        mask = torch.ones_like(t)
        if k != 0:
            values, indices = torch.topk(t, k, largest=False)
            loss2 = 2 * torch.log(torch.exp(values) + 1) - values
            sum_loss2 = torch.sum(loss2)
            mask[indices] = 0

        loss1 = torch.log(torch.exp(-t) + 1)
        loss1 = loss1 * mask
        res = (torch.sum(loss1) + self.lossScale1 * sum_loss2) / outputLen
        return res

class OvrSigmoidLoss(nn.Module):
    def forward(self, output, target):
        device = output.device
        classes = torch.tensor(range(output.shape[1])).reshape(1, -1).to(device)
        boolMatrix = target.view(-1, 1) != classes
        symbolMatrix = torch.ones(output.shape[0], output.shape[1]).to(device)
        symbolMatrix[boolMatrix] = -1

        outputNum = output.shape[0] * output.shape[1]
        assert outputNum != 0
        sigmoidLoss = 2 / (torch.exp(2 * output.mul(symbolMatrix))+1)
        res = sigmoidLoss.sum() / outputNum
        return res

class LogisticAndSigmoidLoss(nn.Module):
    def __init__(self, lossScale = 1, **kwargs):
        super(LogisticAndSigmoidLoss, self).__init__()
        self.lossScale = lossScale
        self.logisticloss = LogisticLoss()
        self.sigmoidloss = SigmoidLoss(self.lossScale)

    def forward(self, output, target):
        if globalVar.getValue('NextLoss') is False:
            return self.logisticloss(output, target)
        return self.sigmoidloss(output, target)

class LogisticAndSavageLoss(nn.Module):
    def __init__(self):
        super(LogisticAndSavageLoss, self).__init__()
        self.logisticloss = LogisticLoss()
        self.savageLoss = SavageLoss()

    def forward(self, output, target):
        if globalVar.getValue('NextLoss') is False:
            return self.logisticloss(output, target)
        return self.savageLoss(output, target)

class LogisticAndRampLoss(nn.Module):
    def __init__(self):
        super(LogisticAndRampLoss, self).__init__()
        self.logisticloss = LogisticLoss()
        self.ramploss = RampLoss()

    def forward(self, output, target):
        if globalVar.getValue('NextLoss') is False:
            return self.logisticloss(output, target)
        return self.ramploss(output, target)

class LogisticAndWelschLoss(nn.Module):
    def __init__(self):
        super(LogisticAndWelschLoss, self).__init__()
        self.logisticloss = LogisticLoss()
        self.welschloss = WelschLoss()

    def forward(self, output, target):
        if globalVar.getValue('NextLoss') is False:
            return self.logisticloss(output, target)
        return self.welschloss(output, target)

class SigmoidLossAndLogistic(nn.Module):
    def __init__(self):
        super(SigmoidLossAndLogistic, self).__init__()
        self.k = 1
        self.logisticloss = LogisticLoss()
        self.sigmoidloss = SigmoidLoss()

    def forward(self, output, target):
        if globalVar.getValue('NextLoss') is False:
            return self.sigmoidloss(output, target, self.k)
        return self.logisticloss(output, target)

def getOneLoss(config):
    if config['criterion'] == 'SigmoidLoss':
        return SigmoidLoss()
    elif config['criterion'] == 'LeastSquareLoss':
        return LeastSquareLoss()
    elif config['criterion'] == 'HingeLoss':
        return HingeLoss()
    elif config['criterion'] == 'SquaredHingeLoss':
        return SquaredHingeLoss()
    elif config['criterion'] == 'LogisticLoss':
        return LogisticLoss()
    elif config['criterion'] == 'FocalLoss':
        return FocalLoss(**config)
    elif config['criterion'] == 'MAILLoss':
        return MAILLoss(**config)
    elif config['criterion'] == 'SOVRLoss':
        return SOVRLoss(**config)
    elif config['criterion'] == 'OvrSigmoidLoss':
        return OvrSigmoidLoss()
    elif config['criterion'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif config['criterion'] == 'LogisticAndSigmoidLoss':
        return LogisticAndSigmoidLoss(**config)
    elif config['criterion'] == 'LogisticAndSavageLoss':
        return LogisticAndSavageLoss()
    elif config['criterion'] == 'LogisticAndRampLoss':
        return LogisticAndRampLoss()
    elif config['criterion'] == 'LogisticAndWelschLoss':
        return LogisticAndWelschLoss()
    elif config['criterion'] == 'SigmoidLossAndLogistic':
        return SigmoidLossAndLogistic()
    assert False

def getLoss(config):
    split = config['split']
    if split not in config['criterion']:
        return getOneLoss(config)
    losses = config['criterion'].split(split)
    temp_config = config
    res = []
    for loss in losses:
        temp_config['criterion'] = loss
        res.append(getOneLoss(temp_config))
    return res
