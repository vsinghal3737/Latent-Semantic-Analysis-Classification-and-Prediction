from flask import Flask, jsonify, request
import json
import os

from PreprocessingData import PreprocessingData
from LSA import LSA
from Prediction import Prediction
from Testing import Testing


app = Flask(__name__)

global data
global lsa
global testing
global prediction
global rareWords
global mapper

global loadedMapper
global loadedRareWords

data = None
lsa = None
testing = None
prediction = None
rareWords = None
mapper = None
loadedMapper = None
loadedRareWords = None


@app.route('/')  # 'http://www.google.com/'
def home():
    return "LSA API"


@app.route('/DataRead', methods=['GET', 'POST'])  # 'http://www.google.com/'
def dataread():
    global data

    if request.method == 'POST':
        request_path = request.json
        data = PreprocessingData(request_path['path'])
    else:
        data = PreprocessingData()

    return 'Data Initialized'


@app.route('/DataSplit', methods=['GET', 'POST'])
def datasplit():
    global data

    if request.method == 'POST':
        request_split_ratio = request.json
        data.DataSplit(request_split_ratio['ratio'])
    else:
        data.DataSplit()

    return 'Data Splitting Done'


@app.route('/DataPrcessing')
def dataprocessing():
    global data
    global mapper
    # return {'data': dir(data)}
    data.WordMapper()
    data.SwapWordByMapper()
    data.CategorizeForEach()
    data.CatergorizeForAll()

    mapper = data.mapper

    return 'Data Processing Done'


@app.route('/TFIDF')
def lsatfidf():
    global lsa
    global data

    lsa = LSA(data)
    lsa.TF_IDF()

    return 'TFIDF Done'


@app.route('/TopWords')
def topword():
    global lsa
    global rareWords
    lsa.TopWords()
    x = {}
    for key in data.keys:
        x[key] = list(lsa.rareWords[key].keys())

    rareWords = lsa.rareWords

    return jsonify({'RareWords': x})


@app.route('/SaveLSA')
def savelsa():
    global rareWords
    global mapper
    with open('rareWords.json', 'w') as file:
        json.dump(rareWords, file)
    with open('mapper.json', 'w') as file:
        json.dump(mapper, file)
    return 'mapper and rareWords saved in ' + os.path.abspath('')


@app.route('/LoadLSA', methods=['GET', 'POST'])
def loadlsa():
    request_path = ''
    if request.method == 'POST':
        request_path = request.json['path'] + '\\'

    global loadedRareWords
    global loadedMapper

    loadedRareWords = None
    with open(request_path + 'rareWords.json', 'r') as file:
        loadedRareWords = json.load(file)

    loadedMapper = None
    with open(request_path + 'mapper.json', 'r') as file:
        loadedMapper = json.load(file)

    try:
        os.path.abspath(request_path)
    except:
        return 'mapper and rareWords are loaded from ' + os.path.abspath('')
    else:
        return 'mapper and rareWords are loaded from ' + os.path.abspath(request_path)


@app.route('/PredictOne', methods=['POST'])
def predictone():
    request_data = request.json

    if not predictionFactory(request_data['loaded']):
        return "Load or Create LSA Model"

    global prediction

    predicted = prediction.predictOne(request_data['data'])

    return jsonify(predicted)


@app.route('/PredictMany', methods=['POST'])
def predictmany():
    request_data = request.json
    if not predictionFactory(request_data['loaded']):
        return "Load or Create LSA Model"

    global prediction
    predicted = prediction.predictMany(request_data['data'])

    return jsonify(predicted)


def predictionFactory(loaded):
    global prediction
    if loaded is True:
        global loadedMapper
        global loadedRareWords
        if loadedMapper is None or loadedRareWords is None:
            return False
        prediction = Prediction(mapper=loadedMapper, rareWords=loadedRareWords)
    else:
        global mapper
        global rareWords
        if mapper is None or rareWords is None:
            return False
        prediction = Prediction(mapper=mapper, rareWords=rareWords)
    return True


@app.route('/Testing')
def testing():
    global prediction
    if not prediction:
        global mapper
        global rareWords
        prediction = Prediction(mapper, rareWords)

    global testing
    predictedOutput = prediction.predictMany(data.TestingData['input'])

    actualOutput = data.TestingData['output']

    testing = Testing(actualOutput, predictedOutput, list(data.keys))

    return 'Setup for Testing Done'


@app.route('/ConfusionMatrix')
def confusionmatrix():
    global testing

    testing.ConfusionMatrix()

    x = {}
    i = 0
    for key in data.keys:
        row1 = ', '.join(list(map(lambda x: str(x), list(testing.confusionMatrix[i][0]))))
        row2 = ', '.join(list(map(lambda x: str(x), list(testing.confusionMatrix[i][1]))))
        x[key] = '[[{}], [{}]]'.format(row1, row2)
        i += 1
    return jsonify({'ConfusionMatrix': x})


@app.route('/Score')
def score():
    global testing

    testing.Score()

    return testing.score


@app.route('/ScoreView')
def scoreview():
    x = 'Label                         Accuracy       Precision      Recall         F1_Score       Support\n'
    x += '_________________________________________________________________________________________________\n'
    for key in data.keys:
        x += '{:30}'.format(key)
        for score in testing.score[key]:
            x += '{:15}'.format(str(round(testing.score[key][score], 2)))
        x += '\n'
    return x


@app.route('/Development')
def Development():
    global data
    global lsa
    global prediction
    global testing

    data = PreprocessingData()
    data.DataSplit(0.3)
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

    x = 'Label                         Accuracy       Precision      Recall         F1_Score       Support\n'
    x += '_________________________________________________________________________________________________\n'
    for key in data.keys:
        x += '{:30}'.format(key)
        for score in testing.score[key]:
            x += '{:15}'.format(str(round(testing.score[key][score], 2)))
        x += '\n'
    return x


if __name__ == '__main__':
    app.run(port=5000, debug=False)
