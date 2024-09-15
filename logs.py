import time
import os

import torch

import numpy as np

import globalVar

class Log():
    def __init__(self, outputFileName, prefixFileName):
        # make dirctory according to current time
        t = time.localtime()
        folder = './Output//{}//{}y_{}m_{}d_{}h_{}m//'.format(prefixFileName, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
        folderExist = os.path.exists(folder)
        if not folderExist:
            os.makedirs(folder)

        file = outputFileName
        csvFile = folder + file + ".csv"
        errorFile = folder + file + "_error.log"
        warningFile = folder + file + "_warning.log"
        normalFile = folder + file + "_normal.log"
        self.csvFilePointer = open(csvFile, mode='w+')
        self.errorFilePointer = open(errorFile, mode='w+')
        self.warningFilePointer = open(warningFile, mode='w+')
        self.normalFilePointer = open(normalFile, mode='w+')

    def errorInfo(self, info):
        print(info)
        self.errorFilePointer.write(info)
        self.errorFilePointer.write("\n")
        self.errorFilePointer.flush()

    def warningInfo(self, info):
        print(info)
        self.warningFilePointer.write(info)
        self.warningFilePointer.write("\n")
        self.warningFilePointer.flush()

    def normalInfo(self, info):
        print(info)
        self.normalFilePointer.write(info)
        self.normalFilePointer.write("\n")
        self.normalFilePointer.flush()

    def writeCSV(self, info):
        self.csvFilePointer.write(info)
        self.csvFilePointer.flush()

    def close(self):
        self.errorFilePointer.close()
        self.warningFilePointer.close()
        self.normalFilePointer.close()
        self.csvFilePointer.close()

# patience = 10 in ERM
class Records():
    def __init__(self, epochs, patience = 5, lossNum = 1, swapRate = 0.8, SaveModelName = None):
        self.lastEpoch = epochs
        self.swapEpoch = int(self.lastEpoch * swapRate)
        self.lastEvalLoss = 99999999
        self.count = patience
        self.patience = patience
        
        self.SaveModelName = SaveModelName
        if self.SaveModelName is not None:
            # make dirctory according to current time
            t = time.localtime()
            folder = './Output//{}y_{}m_{}d_{}h_{}m//'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)
            folderExist = os.path.exists(folder)
            if not folderExist:
                os.makedirs(folder)
            self.SaveModelName = folder + SaveModelName

        globalVar.setValue('NextLoss', False if lossNum == 2 else True)
        self.resRecords = {'totalEpoch': 0, 'swapEpoch': self.lastEpoch, 'evalLoss': np.inf, 'evalAccuracy': 0.0, 
        'oracleLoss': np.inf,'oracleAccuracy': 0.0,
        'lastLoss': np.inf,'lastAccuracy': 0.0}

    def update_direct(self, prefix:str, remained_record_dict:dict=None):
        if remained_record_dict is not None:
            for k, v in remained_record_dict.items():
                self.resRecords[prefix + str(k)] = v

    def update(self, epoch, evalLoss, testLoss, testAccuracy, eval_tolerance = 1e-5, remained_record_dict:dict=None):
        self.resRecords['totalEpoch'] = epoch
        if self.lastEvalLoss > evalLoss:
            self.count = self.patience
            self.lastEvalLoss = evalLoss
            self.resRecords['evalLoss'] = testLoss
            self.resRecords['evalAccuracy'] = testAccuracy
            self.update_direct('eval_', remained_record_dict)
            if self.SaveModelName is not None:
                model = globalVar.getValue('Model')
                torch.save(model.state_dict(), self.SaveModelName)
        else:
            self.count -= 1
        
        if self.resRecords['oracleLoss'] > testLoss:
            self.resRecords['oracleLoss'] = testLoss
            self.resRecords['oracleAccuracy'] = testAccuracy
            self.update_direct('oracle_', remained_record_dict)

        if globalVar.getValue('NextLoss') is False and (self.count <= 0 or epoch > self.swapEpoch):
            self.resRecords['swapEpoch'] = epoch
            self.resRecords['oracleLoss'] = np.inf
            self.lastEvalLoss = np.inf
            self.count = self.patience
            globalVar.setValue('NextLoss', True)
        
        if evalLoss <= eval_tolerance:
            self.count = self.patience if  globalVar.getValue('NextLoss') is False else 0

        if self.lastEpoch == epoch or self.count <= 0:
            self.resRecords['lastLoss'] = testLoss
            self.resRecords['lastAccuracy'] = testAccuracy
            self.update_direct('last_', remained_record_dict)

    def getRecords(self):
        return self.resRecords

    def EarlyStop(self):
        return self.count <= 0
