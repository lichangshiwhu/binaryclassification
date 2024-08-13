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
        
        adv_evaluateDict = algorithm.evaluate(testLoader, is_adversarial=True)
        nat_evaluateDict = algorithm.evaluate(testLoader, is_adversarial=False)
        log.warningInfo('Epoch:{}, Train loss: {:.6f}, Adv Val Loss {:.6f}, Adv  Val acc {:.4f}, Nat Val Loss {:.6f}, Nat Val acc {:.4f}, NextLoss:{}'.format(
            epoch, loss, adv_evaluateDict['val_loss'], adv_evaluateDict['val_accuracy'],\
                  nat_evaluateDict['val_loss'], nat_evaluateDict['val_accuracy'], globalVar.getValue('NextLoss')))
        #  the nat_val_loss is ther loss under the nautral validation set, may be useful in the furture
        remained_record_dict = {'nat_val_loss': nat_evaluateDict['val_loss'], 'nat_loss':nat_evaluateDict['loss'], 'nat_accuracy':nat_evaluateDict['accuracy']}
        records.update(epoch, adv_evaluateDict['val_loss'], adv_evaluateDict['loss'], adv_evaluateDict['accuracy'], remained_record_dict=remained_record_dict)

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
        log.warningInfo("============== repeat: {} , adv_alpha: {}, adv_norm: {} =============="
                        .format(rep, parameter["alpha"], parameter["norm"]))
        recordsDict = train(parameter, log)

        prefixes = parametercombinations.getPrefix()
        log.writeCSV("{},".format(rep))
        for prefix in prefixes:
            log.writeCSV("{},".format(parameter[prefix]))
        log.writeCSV("{}, {}, {},{},{},{},{},{},".format(
        recordsDict['totalEpoch'], recordsDict['swapEpoch'],
        recordsDict['evalLoss'], recordsDict['evalAccuracy'], 
        recordsDict['oracleLoss'], recordsDict['oracleAccuracy'], 
        recordsDict['lastLoss'], recordsDict['lastAccuracy']))
        log.writeCSV("{},{},{}, {},{},{}, {},{},{}\n".format(
        recordsDict['eval_nat_val_loss'], recordsDict['eval_nat_val_loss'], recordsDict['eval_nat_accuracy'],
        recordsDict['oracle_nat_val_loss'], recordsDict['oracle_nat_val_loss'], recordsDict['oracle_nat_accuracy'],
        recordsDict['last_nat_val_loss'], recordsDict['last_nat_val_loss'], recordsDict['last_nat_accuracy']
        ))
    log.close()
