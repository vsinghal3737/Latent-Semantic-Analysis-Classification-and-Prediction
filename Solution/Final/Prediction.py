class Prediction:
    def __init__(self, mapper=None, rareWords=None):
        self.rareWords = rareWords
        self.mapper = mapper

    def predictOne(self, data):
        # WordSwapperWithMapper
        data = set(filter(lambda val: val is not None, map(lambda x: self.mapper[x][1] if x in self.mapper else None, data.split(' '))))

        # CommonWordCheck
        hitCount = {key: 0 for key in self.rareWords.keys()}

        for key in self.rareWords.keys():
            for i in data:
                if i in self.rareWords[key]:
                    hitCount[key] += 1

        return max(hitCount.items(), key=lambda x: x[1])[0]

    def predictMany(self, dataList):
        output = []
        for i in range(len(dataList)):
            output.append(self.predictOne(dataList[i]))
        return output
