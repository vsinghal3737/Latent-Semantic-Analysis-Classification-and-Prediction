import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split


class PreprocessingData:
    def __init__(self, path='shuffled-full-set-hashed.csv'):
        self.TrainingData = pd.read_csv(path, header=None)
        self.TestingData = pd.DataFrame()

        self.TrainingData.columns = ['output', 'input']
        self.TrainingData.dropna(inplace=True)

        self.mapper = {}
        self.docForEach = {}
        self.keys = []
        self.lenForEach = {}
        self.docForAll = {}

    def DataSplit(self, testingRatio=0.3):
        inputTrain, inputTest, outputTrain, outputTest = \
            train_test_split(
                np.array(self.TrainingData['input']),
                np.array(self.TrainingData['output']),
                test_size=testingRatio
            )
        self.TrainingData = pd.DataFrame(zip(outputTrain, inputTrain), columns=['output', 'input'])
        self.TestingData = pd.DataFrame(zip(outputTest, inputTest), columns=['output', 'input'])

    def WordMapper(self):
        # Maper every unique word to a unique ID
        ID = 0
        for i in range(len(self.TrainingData)):
            for word in self.TrainingData.iloc[i][1].split(' '):
                if word not in self.mapper:
                    self.mapper[word] = (word, ID)
                    ID += 1

    def SwapWordByMapper(self):
        # Replace words in DataFrame to mapper ID
        temp = []
        for i in range(len(self.TrainingData)):
            for word in self.TrainingData.iloc[i][1].split(' '):
                temp.append(self.mapper[word][1])
            self.TrainingData.iloc[i][1] = temp[:]
            temp = []

    def CategorizeForEach(self):
        for i in range(len(self.TrainingData)):
            if self.TrainingData.iloc[i][0] in self.docForEach:
                self.docForEach[self.TrainingData.iloc[i][0]].extend(self.TrainingData.iloc[i][1][:])
            else:
                self.docForEach[self.TrainingData.iloc[i][0]] = self.TrainingData.iloc[i][1][:]
        self.keys = self.docForEach.keys()

        temp = {}
        for key in self.docForEach:
            self.lenForEach[key] = len(self.docForEach[key])
            for word in self.docForEach[key]:
                if word in temp:
                    temp[word] += 1
                else:
                    temp[word] = 1
            self.docForEach[key] = deepcopy(temp)
            temp = {}

    def CatergorizeForAll(self):
        self.docForAll = {key: {i: 0 for i in range(len(self.mapper))} for key in self.docForEach.keys()}
        for key in self.docForAll:
            for word in self.docForAll[key]:
                if word in self.docForEach[key]:
                    self.docForAll[key][word] = 1

        self.TrainingData = pd.DataFrame(data=[self.docForAll[key] for key in self.keys], index=self.keys)
