import pandas as pd
import math


class LSA:
    def __init__(self, data):
        self.data = data
        self.DataFrameTFIDF = None
        self.rareWords = {}

    def TF_IDF(self):
        tfForEach = {key: self._computeTF(self.data.docForEach[key], self.data.lenForEach[key]) for key in self.data.keys}
        idfForAllWords = self._computeIDF([self.data.docForAll[key] for key in self.data.keys])

        tfidfForEach = {key: self._computeTFIDF(tfForEach[key], idfForAllWords) for key in self.data.keys}

        self.DataFrameTFIDF = pd.DataFrame(data=[tfidfForEach[key] for key in self.data.keys], index=self.data.keys)
        self.DataFrameTFIDF.dropna(axis=1, how='all', inplace=True)
        self.DataFrameTFIDF.drop(self.DataFrameTFIDF.columns[self.DataFrameTFIDF.iloc[-1, :] == 0], axis=1, inplace=True)

    def TopWords(self, n=100):
        self.rareWords = {
            key: dict.fromkeys(
                map(
                    lambda x: x[0],
                    sorted(
                        filter(lambda x: x[1] != 0, self.DataFrameTFIDF.loc[key].dropna().items()),
                        key=lambda x: x[1],
                        reverse=True
                    )[:n]
                ),
                None
            ) for key in self.data.keys
        }

    def _computeTF(self, wordDict, length):
        return {word: round(count / length, 5) for word, count in wordDict.items()}

    def _computeIDF(self, docList):
        idfDict = {}
        N = len(docList)

        # counts the number of documents that contains word w
        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for doc in docList:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1

        # divide N by denominator above, take the log of that
        for word, val in idfDict.items():
            idfDict[word] = math.log(N / float(val))

        return idfDict

    def _computeTFIDF(self, tf, idfs):
        tfidf = {}
        for word, val in tf.items():
            tfidf[word] = val * idfs[word]
        return tfidf
