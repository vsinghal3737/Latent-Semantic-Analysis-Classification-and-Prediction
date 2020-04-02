class Prediction:
    def __init__(self, mapper=None, rareWords=None):
        self.rareWords = rareWords
        self.mapper = mapper

    def predictOne(self, data, confidence=False):

        # WordSwapperWithMapper
        data = set(
            filter(
                lambda val: val is not None,
                map(
                    lambda x: self.mapper[x][1] if x in self.mapper else None,
                    data.split(' ')
                )
            )
        )

        # Prediction
        hitCount = {key: 0 for key in self.rareWords.keys()}

        for key in self.rareWords.keys():
            for i in data:
                x = str(i) if type(list(self.rareWords['BILL'].keys())[0]) == str else i
                if x in self.rareWords[key]:
                    hitCount[key] += 1
        if not confidence:
            return sorted(hitCount.items(), key=lambda x: x[1], reverse=True)[0][0]

        hitCountSorted = sorted(hitCount.items(), key=lambda x: x[1], reverse=True)
        prediction = hitCountSorted.pop(0)

        # Confidence
        predictedVal = list(map(lambda x: x[1], hitCountSorted))
        average = sum(predictedVal) / len(predictedVal)
        confidence = prediction[1] / (prediction[1] + average)

        return {'prediction': prediction[0], 'confidence': round(confidence, 2)}

    def predictMany(self, dataList, confidence=False):
        return [self.predictOne(dataList[i], confidence) for i in range(len(dataList))]
