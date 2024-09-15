
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import numpy as np

from model import getModel
from losses import getLoss
import globalVar

class ERM(torch.nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, config):
        super(ERM, self).__init__()
        modelClass = getModel(config['modelName'])
        model = modelClass(**config)
        self.device = config['device']
        self.model = model.to(self.device)
        globalVar.setValue('Model', self.model)
        self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'] , momentum=config['momentum'])
        self.criterion = getLoss(config)

    def update(self, x, y, **kwargs):
        classifier = self.predict(x).squeeze(dim=1)
        loss = self.criterion(classifier, y)

        self.optimizer.zero_grad()
        loss.backward()
        # print(nn.Sequential(*list(self.model.children()))[0][8].weight.grad)

        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        classifier = self.model(x)
        return classifier

    def evaluate(self, testLoader, validRate = 0.2):
        with torch.no_grad():
            test_loss = 0
            correct = 0
            dataTotal = 0
            val_loss = 0
            val_accuracy = 0

            validTimes = int(len(testLoader) * validRate)

            for data, target in testLoader:
                dataLen = len(data)
                data, target = (data.to(self.device)),(target.to(self.device))
                output = self.predict(data).squeeze(dim=1)
                # sum up batch loss
                test_loss += self.criterion(output, target).item() * dataLen
                # get the index of the max
                pred = self.getLabels(output)
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                dataTotal += dataLen

                validTimes -= 1
                if validTimes == 0:
                    val_loss = test_loss / dataTotal
                    val_accuracy = 100. * correct / dataTotal

            test_loss /= dataTotal
            test_accuracy = 100. * correct / dataTotal
            return {"loss": test_loss, "accuracy": test_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy}

    def getLabels(self, output):
        if len(output.shape) == 1 or output.shape[1] == 1:
            # for binary classification means $f(x, y) \rightarrow R$
            pred = output
            pred[pred > 0] = 1
            pred[pred <= 0] = -1
        else:
            pred = output.max(1, keepdim=True)[1]
        return pred



class PGD(ERM):
    def __init__(self, config):
        super(PGD, self).__init__(config)
        # <= 10/255.0
        # assert(2/255.0 <= config['epsilon'])
        assert(config['norm'] in [2, 'inf', np.inf])
        # epsilon, magnitude of perturbation, make sure to normalize to 0-1 range
        self.eps = config['epsilon']
        # step size
        self.alpha = config['alpha']
        # l2 or linf
        self.norm = config['norm']
        #iterations
        self.iterations = config['iterations']

    def update(self, x, y, **kwargs):
        x = self.adversarialSamples(x, y)
        classifier = self.predict(x).squeeze(dim=1)
        loss = self.criterion(classifier, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def evaluate(self, testLoader, is_adversarial = True, validRate = 0.2):
        test_loss = 0
        correct = 0
        dataTotal = 0
        val_loss = 0
        val_accuracy = 0

        validTimes = int(len(testLoader) * validRate)

        for data, target in testLoader:
            dataLen = len(data)
            data, target = (data.to(self.device)),(target.to(self.device))
            if is_adversarial:
                data = self.adversarialSamples(data, target)
            with torch.no_grad():
                output = self.predict(data).squeeze(dim=1)
                # sum up batch loss
                test_loss += self.criterion(output, target).item() * dataLen
                # get the index of the max
                pred = self.getLabels(output)
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                dataTotal += dataLen

            validTimes -= 1
            if validTimes == 0:
                val_loss = test_loss / dataTotal
                val_accuracy = 100. * correct / dataTotal

        test_loss /= dataTotal
        test_accuracy = 100. * correct / dataTotal
        return {"loss": test_loss, "accuracy": test_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy}

    def adversarialSamples(self, images, labels):
        if self.alpha == 0:
            return images
        # applies PGD to a batch of images

        adv = images.clone().detach().requires_grad_(True).to(self.device)

        # run for desired number of iterations
        for i in range(self.iterations):
            _adv = adv.clone().detach().requires_grad_(True)

            # predict on current perturbation + input
            outputs = self.model(_adv).squeeze(dim=1)

            # compute classification criterion
            self.model.zero_grad()
            cost = self.criterion(outputs, labels)

            # calculate gradient with respect to the input
            cost.backward()
            grad = _adv.grad

            # normalize gradient into lp ball
            if self.norm in ["inf", np.inf]:
                grad = grad.sign()
            elif self.norm == 2:
                if len(grad.shape) == 4:
                    sum_grad2 = torch.sum(grad * grad, dim=(1,2,3), keepdim=True)
                elif len(grad.shape) == 2:
                    sum_grad2 = torch.sum(grad * grad, dim=(1), keepdim=True)
                else:
                    assert False
                grad = grad / (torch.sqrt(sum_grad2) + 10e-8)

            assert(images.shape == grad.shape)

            # take step in direction of gradient and apply to current example
            adv = adv + grad * self.alpha

            # project current example back onto Lp ball
            if self.norm in ["inf", np.inf]:
                adv = torch.max(torch.min(adv, images + self.eps), images - self.eps)
            elif self.norm == 2:
                d = adv - images
                mask = self.eps >= d.view(d.shape[0], -1).norm(self.norm, dim=1)
                scale = d.view(d.shape[0], -1).norm(self.norm, dim=1)
                scale[mask] = self.eps
                d *= self.eps / scale.view(-1, 1, 1, 1)
                adv = images + d

            # clamp into 0-1 range
            adv = adv.clamp(0.0, 1.0)
        self.model.zero_grad()
        # return adversarial example
        return adv.detach()


def getAlgorithms(config):
    if config['algorithm'] == 'ERM':
        return ERM
    elif config['algorithm'] == 'PGD':
        return PGD
