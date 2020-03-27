from LSA import LSA
from Testing import Testing
from PreprocessingData import PreprocessingData
from Prediction import Prediction

# path = '..\\shuffled-full-set-hashed.csv'
# data = PreprocessingData(path)

data = PreprocessingData()
data.DataSplit()
data.WordMapper()
data.SwapWordByMapper()
data.CategorizeForEach()
data.CatergorizeForAll()

lsa = LSA(data)
lsa.TF_IDF()
lsa.TopWords()


prediction = Prediction(lsa, data)
predictedOutput = prediction.predictMany(data.TestingData['input'])

actualOutput = data.TestingData['output']

testing = Testing(actualOutput, predictedOutput, list(data.keys))
testing.ConfusionMatrix()
testing.Score()

print('Lable                         Accuracy       Precision      Recall         F1_Score       Support')
print('_________________________________________________________________________________________________')
for key in data.keys:
    print('{:30}'.format(key), end='')
    for score in testing.score[key]:
        print('{:15}'.format(str(round(testing.score[key][score], 2))), end='')
    print()
temp = 0
for key in data.keys:
    temp += testing.score[key]['Support']
