from flask import Flask, jsonify, request

from PreprocessingData import PreprocessingData
from LSA import LSA
from Prediction import Prediction
from Testing import Testing


app = Flask(__name__)

global data
global lsa
global testing
global prediction

data = None
lsa = None
testing = None
prediction = None


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
    # return {'data': dir(data)}
    data.WordMapper()
    data.SwapWordByMapper()
    data.CategorizeForEach()
    data.CatergorizeForAll()
    return 'Data Processing Done'


@app.route('/TFIDF')
def lsatfidf():
    global lsa
    global data

    lsa = LSA(data)
    lsa.TF_IDF()

    return 'TFIDF Done'
    # return {'lsa': str(type(lsa))}


@app.route('/TopWords')
def topword():
    global lsa
    lsa.TopWords()
    x = {}
    for key in data.keys:
        x[key] = list(lsa.rareWords[key].keys())

    return jsonify({'RareWords': x})


@app.route('/PredictOne', methods=['POST'])
def predictone():
    global lsa
    global data
    global prediction

    # if not prediction:
    prediction = Prediction(lsa, data)
    request_new_data = request.get_json()
    request_new_data = request_new_data['data']
    # Input = numpy.array(list(map(lambda x: x[1], request_new_data.items())))
    predicted = prediction.predictOne(request_new_data)

    return jsonify({"prediction": predicted})


@app.route('/PredictMany', methods=['POST'])
def predictmany():
    global lsa
    global data
    global prediction

    # if not prediction:
    prediction = Prediction(lsa, data)
    request_new_data = request.get_json()
    request_new_data = request_new_data['data']
    # Input = numpy.array(list(map(lambda x: x[1], request_new_data.items())))
    predicted = prediction.predictMany(request_new_data)

    return jsonify({"prediction": predicted})


@app.route('/Testing')
def testing():
    global lsa
    global data
    global testing
    global prediction
    # if not prediction:
    prediction = Prediction(lsa, data)

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
    app.run(port=5000, debug=True)
