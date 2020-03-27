import numpy as np


class Prediction:
    def __init__(self, lsa, data):
        self.lsa = lsa
        self.data = data

    def predictOne(self, data):
        # WordSwapperWithMapper

        data = set(filter(lambda val: val is not None, map(lambda x: self.data.mapper[x][1] if x in self.data.mapper else None, data.split(' '))))

        # CommonWordCheck
        hitCount = {key: 0 for key in self.data.keys}

        for key in self.data.keys:
            for i in data:
                if i in self.lsa.rareWords[key]:
                    hitCount[key] += 1

        return max(hitCount.items(), key=lambda x: x[1])[0]

    def predictMany(self, dataList):
        output = []
        for i in range(len(dataList)):
            output.append(self.predictOne(dataList[i]))
        return np.array(output)
