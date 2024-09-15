import random
import os
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms

from PIL import Image
import concurrent.futures 
from concurrent.futures import ThreadPoolExecutor  

# sklearn
import sklearn.datasets as skdatasets
from sklearn.preprocessing import MinMaxScaler


#numpy
import numpy as np


# def load_images_in_parallel(image_file_tuples, num_threads=4):  
#     images = []
#     labels = []
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:  
#         futures = []
#         for image_file, label in image_file_tuples:
#             future = executor.submit(read_image, image_file, label)  
#             futures.append(future)

#         for future in concurrent.futures.as_completed(futures):  
#             img, label = future.result()  
#             if img is not None:  
#                 images.append(img)
#                 labels.append(label)  
#     return images, labels


class AlldataLoader():
    def __init__(self, batchSize = 16):
        super(AlldataLoader, self).__init__()
        self.batchSize = batchSize

    def getLoader(self):
        # Data Loader (Input Pipeline)

        trainLoader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.batchSize,
                                                shuffle=True)

        testLoader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.batchSize,
                                                shuffle=False)
        return trainLoader, testLoader

    def generateMislabeledData(self, MislabeledNoise = 0):
        if MislabeledNoise != 0:
            noiseDataLen = int(len(self.train_dataset) * MislabeledNoise)
            index = [i for i in range(len(self.train_dataset))]
            random.shuffle(index)
            self.train_dataset.y[index[0:noiseDataLen]] = -1 * self.train_dataset.y[index[0:noiseDataLen]]


class MNISTLoader(AlldataLoader):
    def __init__(self, batchSize = 16):
        super(MNISTLoader, self).__init__(batchSize)
        self.batchSize = batchSize
        # MNIST Dataset
        self.train_dataset = datasets.MNIST(root='./mnist_data/',
                                    train = True ,
                                    transform = transforms.ToTensor(),
                                    download=True)

        self.test_dataset = datasets.MNIST(root='./mnist_data/',
                                    train=False,
                                    transform=transforms.ToTensor())

class BinaryMNISTLoader(AlldataLoader):
    def __init__(self, batchSize = 16, noise=0):
        super(BinaryMNISTLoader, self).__init__(batchSize)
        self.batchSize = batchSize
        # MNIST Dataset
        self.train_dataset = datasets.MNIST(root='./mnist_data/',
                                    train = True ,
                                    transform = transforms.ToTensor(),
                                    download=True)

        self.train_dataset.targets[self.train_dataset.targets < 5] = -1
        self.train_dataset.targets[self.train_dataset.targets >= 5] = 1

        if noise != 0:
            noiseDataLen = int(len(self.train_dataset.targets) * noise)
            index = [i for i in range(len(self.train_dataset.targets))]
            random.shuffle(index)
            self.train_dataset.targets[index[0:noiseDataLen]] = -1 * self.train_dataset.targets[index[0:noiseDataLen]]

        self.test_dataset = datasets.MNIST(root='./mnist_data/',
                                    train=False,
                                    transform=transforms.ToTensor())
        self.test_dataset.targets[self.test_dataset.targets < 5] = -1
        self.test_dataset.targets[self.test_dataset.targets >= 5] = 1

class ImageFloderLoader(AlldataLoader):
    def  __init__(self, batchSize=16, inputSize = 224*224, MisLabeledNoise = 0, seed = 0, imagefolder = '', valimagefolder = ''):
        super(ImageFloderLoader, self).__init__(batchSize)
        self.train_transform = transforms.Compose([  
        transforms.Resize((256, 256)),  
        transforms.CenterCrop((224, 224)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        self.eval_transform = transforms.Compose([  
        transforms.CenterCrop((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        # cannot identify image file: Dog/11702.jpg, Cat/666.jpg 
        # find and delete above pictrures 
        self.seed = seed
        x, y = self.get_imagepathandlabels(imagefolder)
        if valimagefolder != '':
            train_x, _train_y = x, y
            test_x, _test_y = self.get_imagepathandlabels(valimagefolder)
        else:
            self.trainSamples = int(0.8 * len(y))
            self.testSamples = len(y) - self.trainSamples
            train_x, _train_y = x[:self.trainSamples], y[:self.trainSamples]
            test_x, _test_y = x[self.trainSamples:], y[self.trainSamples:]

        train_x, train_y = self.get_imageandlabels(imagepaths=train_x, labels=_train_y, transform=self.train_transform)
        test_x, test_y = self.get_imageandlabels(imagepaths=test_x, labels=_test_y, transform=self.eval_transform)
        self.train_dataset = makeClassifictionDataset(train_x, train_y)
        self.test_dataset = makeClassifictionDataset(test_x, test_y)
        self.generateMislabeledData(MisLabeledNoise)

    def shuffle_together(self, a, b):  
        indices = list(range(len(a)))  
        
        random.shuffle(indices)  
        
        a_shuffled = [a[i] for i in indices]  
        b_shuffled = [b[i] for i in indices]  
        
        return a_shuffled, b_shuffled  

    def get_imagepathandlabels(self, imagefloder):
        dataset = datasets.ImageFolder(imagefloder)
        imagepaths = []
        labels = []
        for path, label in dataset.samples:
            imagepaths.append(path)
            if label == 1:
                labels.append(1)
            elif label == 0:
                labels.append(-1)
            else:
                import sys
                print("Error labels.")
                sys.exit(0)
        imagepaths, labels = self.shuffle_together(imagepaths, labels)
        return imagepaths, labels

    def get_imageandlabels(self, imagepaths, labels, transform):
        images = []
        _labels = []
        for path, label in list(zip(imagepaths, labels)):
            _x, _y  = self.read_image(path, label, transform)
            if _x is not None:
                images.append(_x)
                _labels.append(_y)
        return images, _labels

    def read_image(self, img_path, label, transform):
        try:  
            img = Image.open(img_path)
            img = transform(img)
            if img is None:  
                raise ValueError(f"cannot find {img_path}")  
            return img, label
        except Exception as e:  
            print(f"Error:{e} when read {img_path}")  
            return None, None

    def generateMislabeledData(self, MislabeledNoise = 0):
        if MislabeledNoise != 0:
            noiseDataLen = int(len(self.train_dataset) * MislabeledNoise)
            index = [i for i in range(len(self.train_dataset))]
            random.shuffle(index)
            indeices = index[0:noiseDataLen]
            for i in indeices:
                self.train_dataset.y[i] = -1 * self.train_dataset.y[i]

class CatAndDogLoader(ImageFloderLoader):
    def __init__(self, batchSize=16, inputSize = 224*224, MisLabeledNoise = 0, seed = 0):
        super(CatAndDogLoader, self).__init__(batchSize=batchSize, inputSize = inputSize, MisLabeledNoise = MisLabeledNoise,
                                               seed = seed, imagefolder = './mydataset/catanddog/kagglecatsanddogs_3367a/PetImages') 

class ShellsorPebblesLoader(ImageFloderLoader):
    def __init__(self, batchSize=16, inputSize = 224*224, MisLabeledNoise = 0, seed = 0):
        super(ShellsorPebblesLoader, self).__init__(batchSize=batchSize, inputSize = inputSize, MisLabeledNoise = MisLabeledNoise,
                                               seed = seed, imagefolder = './mydataset/archive') 

class CactusAerialPhotosLoader(ImageFloderLoader):
    def __init__(self, batchSize=16, inputSize = 224*224, MisLabeledNoise = 0, seed = 0):
        super(CactusAerialPhotosLoader, self).__init__(batchSize=batchSize, inputSize = inputSize, MisLabeledNoise = MisLabeledNoise,
                                               seed = seed, imagefolder = './mydataset/CactusAerialPhotos/training_set/training_set',
                                                valimagefolder = './mydataset/CactusAerialPhotos/validation_set/validation_set') 

class makeClassifictionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(makeClassifictionDataset, self).__init__()
        if isinstance(x, np.ndarray):
            self.nSamples = x.shape[0]
            self.x = [_x.astype(np.float32) for _x in x]
            self.y = [_y.astype(np.float32) for _y in y]
        else:
            self.nSamples = len(y)
            self.x = x
            self.y = y
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.nSamples


class makeClassifictionLoader(AlldataLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 20, seed = 0, outputSize = 1, noise = 0, MisLabeledNoise = 0):
        super(makeClassifictionLoader, self).__init__(batchSize)
        self.batchSize = batchSize
        self.trainSamples = trainSamples
        self.nSamples = trainSamples + testSamples
        self.nFeatures = nFeatures
        self.seed = seed
        self.noise = noise

        self.x, self.y = self.getMakeClassification()
        scale = MinMaxScaler()
        self.x = scale.fit_transform(self.x)
        if outputSize == 1:
            self.y[self.y == 0] = -1

        # x1 = np.array([[1, 0.05],[0.99, 0.04]])
        # y1 = np.array([-1, -1])
        # self.x = np.concatenate((x1, self.x))
        # self.y = np.concatenate((y1, self.y))

        self.train_dataset = makeClassifictionDataset(self.x[:self.trainSamples, :], self.y[:self.trainSamples])

        self.generateMislabeledData(MisLabeledNoise)
        self.test_dataset = makeClassifictionDataset(self.x[self.trainSamples:, :], self.y[self.trainSamples:])

    def getMakeClassification(self):
        pass

class makeLinearClassifictionLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 20, seed = 0, outputSize = 1, noise = 0):
        super(makeLinearClassifictionLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, noise)

    def getMakeClassification(self):
        return skdatasets.make_classification(n_samples = self.nSamples, n_features=self.nFeatures, random_state=self.seed)

class makeCircleLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 2, seed = 0, outputSize = 1, noise = 0, MisLabeledNoise = 0):
        assert nFeatures == 2
        super(makeCircleLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, noise, MisLabeledNoise)

    def getMakeClassification(self):
        return skdatasets.make_circles(n_samples = self.nSamples, noise = self.noise)


class makeMoonLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 2, seed = 0, outputSize = 1, noise = 0, MisLabeledNoise = 0):
        assert nFeatures == 2
        super(makeMoonLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, noise, MisLabeledNoise)

    def getMakeClassification(self):
        X, Y = skdatasets.make_moons(n_samples = self.nSamples, noise = self.noise)
        return X, Y

class breastCancerLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 2, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 30
        super(breastCancerLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        data = skdatasets.load_breast_cancer()
        return data.data, data.target

class fetchLfwPairsLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 2, seed = 0, outputSize = 1, noise = 0):
        assert nFeatures == 5828
        super(fetchLfwPairsLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, noise)

    def getMakeClassification(self):
        train_lfw_pairs = skdatasets.fetch_lfw_pairs(subset = 'train', data_home = './scikit_learn_data')
        test_lfw_pairs = skdatasets.fetch_lfw_pairs(subset = 'test', data_home = './scikit_learn_data')
        trainData = train_lfw_pairs.data
        trainLabel = train_lfw_pairs.target
        testData = test_lfw_pairs.data
        testLabel = test_lfw_pairs.target
        data = np.concatenate((trainData, testData), axis = 0)
        label = np.concatenate((trainLabel, testLabel), axis = 0)
        self.trainSamples = trainData.shape[0]
        return data, label

class ionosphereLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 2, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 34
        super(ionosphereLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/ionosphere.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        labelstr = np.genfromtxt(fileName, dtype=str, delimiter=',')[:, -1]
        
        label = np.zeros_like(labelstr)
        label[labelstr == 'g'] = 1.0
        label[labelstr == 'b'] = -1.0
        label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class raisinLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 100, testSamples=20, nFeatures = 7, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 7
        super(raisinLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/Raisin_Dataset.CSV"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:, -1]
        
        # label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class germanLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 1000, testSamples=20, nFeatures = 24, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 24
        super(germanLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/german.data-numeric"

        data = np.loadtxt(fileName)[:, :-1]
        label = np.loadtxt(fileName)[:, -1]
        label[label == 2] = -1

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class spambaseLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 4601, testSamples=57, nFeatures = 24, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 57
        super(spambaseLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/spambase.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:, -1]
        label[label == 0] = -1

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class docccLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 23
        super(docccLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/default_of_credit_card_clients.CSV"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:, -1]
        label[label == 0] = -1

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class magicLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 10
        super(magicLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/magic04.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        labelstr = np.genfromtxt(fileName, dtype=str, delimiter=',')[:, -1]
        
        label = np.zeros_like(labelstr)
        label[labelstr == 'g'] = 1.0
        label[labelstr == 'h'] = -1.0
        label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class adLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 1558
        super(adLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/ad.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        labelstr = np.genfromtxt(fileName, dtype=str, delimiter=',')[:, -1]
        
        label = np.zeros_like(labelstr)
        label[labelstr == 'ad.'] = 1.0
        label[labelstr == 'nonad.'] = -1.0
        label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class hepatitisLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 19
        super(hepatitisLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/hepatitis.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:, -1]
        
        label[label == 2] = -1
        label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

#Congressional Voting Records
class houseVotesLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 16
        super(houseVotesLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/house-votes-84.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, 1:]
        labelstr = np.genfromtxt(fileName, dtype=str, delimiter=',')[:,0]
        
        label = np.zeros_like(labelstr)
        label[labelstr == 'republica0'] = 1.0
        label[labelstr == 'democrat'] = -1.0
        label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class sonarLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 60
        super(sonarLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/sonar.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        labelstr = np.genfromtxt(fileName, dtype=str, delimiter=',')[:,-1]
        
        label = np.zeros_like(labelstr)
        label[labelstr == 'R'] = 1.0
        label[labelstr == 'M'] = -1.0
        label = label.astype(float)

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class ticTacToeLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 9
        super(ticTacToeLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/tic-tac-toe.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:,-1]
        
        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

# Statlog (Australian Credit Approval)
class statlogLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 14
        super(statlogLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/australian.dat"

        data = np.genfromtxt(fileName, delimiter=' ')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=' ')[:,-1]
        
        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class banknoteAuthenticationLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 4
        super(banknoteAuthenticationLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/data_banknote_authentication.txt"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:,-1]
        label[label == 0] = -1

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

# Early stage diabetes risk prediction dataset.
class earlyStageDiabetesRiskPredictionLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 16
        super(earlyStageDiabetesRiskPredictionLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/diabetes_data_upload.csv"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:,-1]

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label
# Horse Colic
class horseColicLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 27
        super(horseColicLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        trainFileName = "./mydataset/horseColic/horse-colic.data"

        trainData = np.genfromtxt(trainFileName, delimiter=' ')[:, :-1]
        trainLabel = np.genfromtxt(trainFileName, delimiter=' ')[:,-1]

        testFileName = "./mydataset/horseColic/horse-colic.test"

        testData = np.genfromtxt(testFileName, delimiter=' ')[:, :-1]
        testLabel = np.genfromtxt(testFileName, delimiter=' ')[:,-1]

        data = np.concatenate([trainData, testData], axis=0)
        label = np.concatenate([trainLabel, testLabel], axis=0)
        return data, label
# Phishing Websites
class phishingWebsitesLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 30
        super(phishingWebsitesLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/phishing+websites/Training Dataset.arff"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:,-1]

        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label
# Haberman's Survival
class habermanSurvivalLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 3
        super(habermanSurvivalLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/haberman+s+survival/haberman.data"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:,-1]

        label[label == 2] = -1
        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label
# Musk (Version 1)
class muskLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 23, seed = 0, outputSize = 1, MisLabeledNoise = 0):
        assert nFeatures == 166
        super(muskLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def getMakeClassification(self):
        fileName = "./mydataset/musk+version+1/clean1.csv"

        data = np.genfromtxt(fileName, delimiter=',')[:, :-1]
        label = np.genfromtxt(fileName, delimiter=',')[:,-1]

        label[label == 2] = -1
        random.seed(self.seed)
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data[index] = data
        label[index] = label
        return data, label

class makeByYeqXLoader(makeClassifictionLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 2, seed = 0, outputSize = 1, MisLabeledNoise = 0, margin=0.1):
        assert nFeatures == 2
        self.margin = margin
        super(makeByYeqXLoader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, 0, MisLabeledNoise)

    def generate_dataset(self, size):
        np.random.seed(self.seed)
        data = np.empty((0, 2))
        label = np.ones(size)

        while data.shape[0] < size:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            
            distance = np.abs(x-y) / np.sqrt(2)
            
            if distance >= self.margin:
                data = np.vstack((data, [x, y]))
                if x < y:
                    label[data.shape[0]-1] = -1
        return data, label

    def getMakeClassification(self):
        data, label = self.generate_dataset(self.nSamples)
        return data, label

class makeCircleX2Y2Loader(makeByYeqXLoader):
    def __init__(self, batchSize = 16, trainSamples = 3000, testSamples=57, nFeatures = 2, seed = 0, outputSize = 1, MisLabeledNoise = 0, margin=0.1):
        assert nFeatures == 2
        super(makeCircleX2Y2Loader, self).__init__(batchSize, trainSamples, testSamples, nFeatures, seed, outputSize, MisLabeledNoise, margin)

    def generate_dataset(self, size):
        np.random.seed(self.seed)
        data = np.empty((0, 2))
        label = np.ones(size)
        r2 = 0.6
        while data.shape[0] < size:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            p = x * x + y * y

            distance = np.abs(2 * (p - r2)) / (np.sqrt(4 * p + 1) + np.sqrt(4 * r2))

            if distance >= self.margin:
                data = np.vstack((data, [x, y]))
                if p < r2:
                    label[data.shape[0]-1] = -1
        return data, label

def getLoader(config):
    if config['loaderName'] == 'MNIST':
        config['inputSize'] = 784
        config['outputSize'] = 10
        mnistloader = MNISTLoader(config['batchSize'])
        return mnistloader.getLoader()
    if config['loaderName'] == 'BinaryMNIST':
        config['inputSize'] = 784
        config['outputSize'] = 1
        mnistloader = BinaryMNISTLoader(config['batchSize'], config['noise'])
        return mnistloader.getLoader()
    elif config['loaderName'] == 'CatAndDog':
        catanddogloader = CatAndDogLoader(batchSize=config['batchSize'], inputSize = 224*224, MisLabeledNoise = config['MisLabeledNoise'], seed = config['datasetSeed'])
        return catanddogloader.getLoader()
    elif config['loaderName'] == 'ShellsorPebbles':
        ShellsorPebblesloader = ShellsorPebblesLoader(batchSize=config['batchSize'], inputSize = 224*224, MisLabeledNoise = config['MisLabeledNoise'], seed = config['datasetSeed'])
        return ShellsorPebblesloader.getLoader()
    elif config['loaderName'] == 'CactusAerialPhotos':
        CactusAerialPhotosloader = CactusAerialPhotosLoader(batchSize=config['batchSize'], inputSize = 224*224, MisLabeledNoise = config['MisLabeledNoise'], seed = config['datasetSeed'])
        return CactusAerialPhotosloader.getLoader()
    elif config['loaderName'] == 'makeLinearClassifiction':
        makelinearclassifictionloader = makeLinearClassifictionLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'])
        return makelinearclassifictionloader.getLoader()
    elif config['loaderName'] == 'makeCircle':
        makecircleloader = makeCircleLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['noise'], config['MisLabeledNoise'])
        return makecircleloader.getLoader()
    elif config['loaderName'] == 'makeMoon':
        makemoonloader = makeMoonLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['noise'], config['MisLabeledNoise'])
        return makemoonloader.getLoader()
    elif config['loaderName'] == 'makeByYeqX':
        makeByYeqXloader = makeByYeqXLoader(batchSize=config['batchSize'], trainSamples=config['trainSamples'], 
        testSamples=config['testSamples'], nFeatures=config['inputSize'], seed=config['datasetSeed'], outputSize=config['outputSize'], MisLabeledNoise=config['MisLabeledNoise'], margin=config['margin'])
        return makeByYeqXloader.getLoader()
    elif config['loaderName'] == 'makeCircleX2Y2':
        makeCircleX2Y2loader = makeCircleX2Y2Loader(batchSize=config['batchSize'], trainSamples=config['trainSamples'], 
        testSamples=config['testSamples'], nFeatures=config['inputSize'], seed=config['datasetSeed'], outputSize=config['outputSize'], MisLabeledNoise=config['MisLabeledNoise'], margin=config['margin'])
        return makeCircleX2Y2loader.getLoader()
    elif config['loaderName'] == 'breastCancer':
        nSamples = 569
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        breastcancerloader = breastCancerLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return breastcancerloader.getLoader()
    elif config['loaderName'] == 'fetchLfwPairs':
        fetchlfwpairsloader = fetchLfwPairsLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'])
        return fetchlfwpairsloader.getLoader()
    elif config['loaderName'] == 'ionosphere':
        nSamples = 351
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        ionosphereloader = ionosphereLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return ionosphereloader.getLoader()
    elif config['loaderName'] == 'raisin':
        nSamples = 900
        config['inputSize'] = 7
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        raisinloader = raisinLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return raisinloader.getLoader()
    elif config['loaderName'] == 'german':
        nSamples = 1000
        config['inputSize'] = 24
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        germanloader = germanLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return germanloader.getLoader()
    elif config['loaderName'] == 'spambase':
        nSamples = 4601
        config['inputSize'] = 57
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        spambaseloader = spambaseLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return spambaseloader.getLoader()
    elif config['loaderName'] == 'doccc':
        nSamples = 30000
        config['inputSize'] = 23
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        docccloader = docccLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return docccloader.getLoader()
    elif config['loaderName'] == 'magic':
        nSamples = 19020
        config['inputSize'] = 10
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        magicloader = magicLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return magicloader.getLoader()
    elif config['loaderName'] == 'ad':
        nSamples = 3279
        config['inputSize'] = 1558
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        adloader = adLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return adloader.getLoader()
    elif config['loaderName'] == 'hepatitis':
        nSamples = 155
        config['inputSize'] = 19
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        hepatitisloader = hepatitisLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return hepatitisloader.getLoader()
    elif config['loaderName'] == 'houseVotes':
        nSamples = 435
        config['inputSize'] = 16
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        houseVotesloader = houseVotesLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return houseVotesloader.getLoader()
    elif config['loaderName'] == 'sonar':
        nSamples = 208
        config['inputSize'] = 60
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        sonarloader = sonarLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return sonarloader.getLoader()
    elif config['loaderName'] == 'ticTacToe':
        nSamples = 958
        config['inputSize'] = 9
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        ticTacToeloader = ticTacToeLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return ticTacToeloader.getLoader()
    elif config['loaderName'] == 'statlog':
        nSamples = 690
        config['inputSize'] = 14
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        statlogloader = statlogLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return statlogloader.getLoader()
    elif config['loaderName'] == 'banknoteAuthentication':
        nSamples = 1372
        config['inputSize'] = 4
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        banknoteAuthenticationloader = banknoteAuthenticationLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return banknoteAuthenticationloader.getLoader()
    elif config['loaderName'] == 'horseColic':
        nSamples = 368
        config['inputSize'] = 27
        config['outputSize'] = 1
        config['trainSamples'] = 300
        config['testSamples'] = 68
        horseColicloader = horseColicLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return horseColicloader.getLoader()
    elif config['loaderName'] == 'phishingWebsites':
        nSamples = 11055
        config['inputSize'] = 30
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        phishingWebsitesloader = phishingWebsitesLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return phishingWebsitesloader.getLoader()
    elif config['loaderName'] == 'habermanSurvival':
        nSamples = 306
        config['inputSize'] = 3
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        habermanSurvivalloader = habermanSurvivalLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return habermanSurvivalloader.getLoader()
    elif config['loaderName'] == 'musk':
        nSamples = 476
        config['inputSize'] = 166
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        muskloader = muskLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return muskloader.getLoader()
    elif config['loaderName'] == 'earlyStageDiabetesRiskPrediction':
        nSamples = 520
        config['inputSize'] = 16
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        earlyStageDiabetesRiskPredictionloader = earlyStageDiabetesRiskPredictionLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return earlyStageDiabetesRiskPredictionloader.getLoader()
    else:
        assert False


def getDataset(config):
    if config['loaderName'] == 'CatAndDog':
        catanddogloader = CatAndDogLoader(config['batchSize'], config['inputSize'])
        return catanddogloader.getDataset()
    elif config['loaderName'] == 'makeCircle':
        makecircleloader = makeCircleLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['noise'], config['MisLabeledNoise'])
        return makecircleloader.getDataset()
    elif config['loaderName'] == 'makeMoon':
        makemoonloader = makeMoonLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['noise'], config['MisLabeledNoise'])
        return makemoonloader.getDataset()
    elif config['loaderName'] == 'breastCancer':
        nSamples = 569
        config['inputSize'] = 30
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        breastcancerloader = breastCancerLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return breastcancerloader.getDataset()
    elif config['loaderName'] == 'ionosphere':
        nSamples = 351
        config['inputSize'] = 34
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        ionosphereloader = ionosphereLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return ionosphereloader.getDataset()
    elif config['loaderName'] == 'houseVotes':
        nSamples = 435
        config['inputSize'] = 16
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        houseVotesloader = houseVotesLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return houseVotesloader.getDataset()
    elif config['loaderName'] == 'statlog':
        nSamples = 690
        config['inputSize'] = 14
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        statlogloader = statlogLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return statlogloader.getDataset()
    elif config['loaderName'] == 'musk':
        nSamples = 476
        config['inputSize'] = 166
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        muskloader = muskLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return muskloader.getDataset()
    elif config['loaderName'] == 'earlyStageDiabetesRiskPrediction':
        nSamples = 520
        config['inputSize'] = 16
        config['outputSize'] = 1
        config['trainSamples'] = int(nSamples * 0.8)
        config['testSamples'] = nSamples - config['trainSamples']
        earlyStageDiabetesRiskPredictionloader = earlyStageDiabetesRiskPredictionLoader(config['batchSize'], config['trainSamples'], 
        config['testSamples'], config['inputSize'], config['datasetSeed'], config['outputSize'], config['MisLabeledNoise'])
        return earlyStageDiabetesRiskPredictionloader.getDataset()
    else:
        assert False
