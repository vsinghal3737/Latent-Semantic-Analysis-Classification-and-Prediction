from sklearn.metrics import multilabel_confusion_matrix


class Testing:
    def __init__(self, actualOutput, predictedOutput, labels):
        self.actualOutput = actualOutput
        self.predictedOutput = predictedOutput
        self.labels = labels

    def ConfusionMatrix(self):
        self.confusionMatrix = multilabel_confusion_matrix(self.actualOutput, self.predictedOutput, labels=self.labels)

    def Score(self):
        self.score = {label: None for label in self.labels}

        support = {}
        for i in self.actualOutput:
            if i in support:
                support[i] += 1
            else:
                support[i] = 1

        for i in range(len(self.confusionMatrix)):
            TP = self.confusionMatrix[i][0][0]
            FN = self.confusionMatrix[i][0][1]
            FP = self.confusionMatrix[i][1][0]
            TN = self.confusionMatrix[i][1][1]
            Accuracy = (TP + TN) / (TP + TN + FP + FN)
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1_Score = 2 * Recall * Precision / (Recall + Precision)
            self.score[self.labels[i]] = {
                'Accuracy': round(Accuracy, 2),
                'Precision': round(Precision, 2),
                'Recall': round(Recall, 2),
                'F1_Score': round(F1_Score, 2),
                'Support': support[self.labels[i]]
            }
