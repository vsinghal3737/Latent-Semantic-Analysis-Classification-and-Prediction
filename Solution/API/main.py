from LSA import LSA
from Testing import Testing
from PreprocessingData import PreprocessingData
from Prediction import Prediction

path = '..\\shuffled-full-set-hashed.csv'

data = PreprocessingData(path)
data.DataSplit()
data.WordMapper()
data.SwapWordByMapper()
data.Categorize()
data.CatergorizeForAll()

lsa = LSA(data)
lsa.TF_IDF()
lsa.TopWords()

prediction = Prediction(lsa, data)

prediction.predictMany(lsa.data.TestingData)

testing = Testing(actualValues, predictedValues)
