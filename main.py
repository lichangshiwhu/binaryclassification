import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

import numpy as np

from model import getModel
from algorithms import getAlgorithms
from loaders import getLoader
from logs import Log, Records
from parameters import parameterCombinations
import globalVar

'''
demends: 
1) input data (training data) must local in [0, 1] for adversarial training.
'''

binaryconfig = {
    # model parameters
    'modelName': 'DNN',
    'activateFunc': nn.ReLU(),
    # is input features
    'inputSize': 2,
    'width': 10000,
    'deep': 2,
    'outputSize':1,
    'tau':1,
    # algorithm parameters
    'algorithm':'ERM',
    'lr': 0.01,
    'momentum': 0.1,
    'criterion':"SigmoidLoss",
    # dataset parameters
    'batchSize': 16,
    'trainSamples':1000,
    'testSamples': 100,
    'datasetSeed':3,
    'loaderName': 'makeMoon',
    'noise': 0,
    # parameters in main
    'epochs': 100,
    # print parameters
    'interval': 2000,
    'repeat':10,
    'file': 'sigmoid_circle_test'
}


def train(config, log):
    train_loader, testLoader = getLoader(config)
    algorithm = getAlgorithms(config)(config)

    SaveModelName = None
    if config.get('SaveModelName') is True:
        SaveModelName = config['criterion'] + '.pth'

    records = Records(config['epochs'], 10, 2 if ('And' in config['criterion']) else 1, SaveModelName = SaveModelName)
    device = config['device']
    for epoch in range(1, config['epochs'] + 1):
        loss = 0
        totalData = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = (data.to(device)), (target.to(device))
            loss += algorithm.update(data, target)['loss'] * len(data)
            totalData += len(data)

        assert totalData != 0
        loss /= totalData
        
        evaluateDict = algorithm.evaluate(testLoader)
        log.warningInfo('Epoch:{}, Train loss: {:.6f}, Val Loss {:.6f}, Val acc {:.4f}, NextLoss:{}'.format(
            epoch, loss, evaluateDict['val_loss'], evaluateDict['val_accuracy'], globalVar.getValue('NextLoss')))

        records.update(epoch, evaluateDict['val_loss'], evaluateDict['loss'], evaluateDict['accuracy'])

        if records.EarlyStop():
            break

    return records.getRecords()

parser = argparse.ArgumentParser()
parser.add_argument('--fileName', '-F', required=True, type=str)
parser.add_argument('--outFileName', '-O', required=True, type=str)
parser.add_argument('--begin', '-B', type=int, default=0)
parser.add_argument('--end', '-E', type=int, default=10)
args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 

for rep in range(args.begin, args.end):
    parametercombinations = parameterCombinations(args.fileName)
    log = Log(args.outFileName + "_" + str(rep + 1), parametercombinations.parameters['loaderName'])
    for parameter in parametercombinations():
        set_seed(rep)
        globalVar._init()
        parameter['datasetSeed'] = rep
        log.warningInfo("============== repeat: {} , width: {}, deep: {}, MisLabeledNoise:{} =============="
                        .format(rep, parameter["width"], parameter["deep"], parameter["MisLabeledNoise"]))
        recordsDict = train(parameter, log)

        prefixes = parametercombinations.getPrefix()
        log.writeCSV("{},".format(rep))
        for prefix in prefixes:
            log.writeCSV("{},".format(parameter[prefix]))
        log.writeCSV("{}, {}, {},{},{},{},{},{}\n".format(
        recordsDict['totalEpoch'], recordsDict['swapEpoch'],
        recordsDict['evalLoss'], recordsDict['evalAccuracy'], 
        recordsDict['oracleLoss'], recordsDict['oracleAccuracy'], 
        recordsDict['lastLoss'], recordsDict['lastAccuracy']))
    log.close()
