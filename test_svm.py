import os
import random

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from loaders import getDataset


config = {
    # is input features
    'inputSize': 2,
    'outputSize':1,
    'tau':1,
    # algorithm parameters
    'algorithm':'ERM',
    'lr': 0.01,
    'momentum': 0.1,
    'criterion':"SigmoidLoss",
    # dataset parameters
    'batchSize': 16,
    'MisLabeledNoise': 0,
    'trainSamples':1000,
    'testSamples': 100,
    'datasetSeed':3,
    'loaderName': 'makeMoon',
    'noise': 0,
}

loaderNames = ['breastCancer', 'ionosphere', 'houseVotes', 'statlog', 'musk', 'earlyStageDiabetesRiskPrediction']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 

for loadername in loaderNames:
    config['loaderName'] = loadername
    sum_acc = 0
    for rep in range(0, 10):
        train_dataset, test_dataset = getDataset(config)
        clf = svm.SVC()
        clf.fit(train_dataset.x, train_dataset.y)

        predY = clf.predict(test_dataset.x)
        acc = accuracy_score(test_dataset.y, predY)
        sum_acc += acc
        print(f"rep = {rep}, acc = {acc}", end = ',')
    print()
    print(f"{loadername}, avg acc = {sum_acc/10}")
