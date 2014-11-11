#! /usr/bin/env python

import json
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

training_Features = []
training_Labels = []
testing_Features = []
testing_Labels = []

numberOfSamples = 1000
#the number of training and testing samples
trainingSamples = int(0.8 * numberOfSamples)
testingSamples = int(0.2 * numberOfSamples)

def Data_Preparation(filename):

    global training_Features
    global training_Labels
    global testing_Features
    global testing_Labels

    features = []
    labels = []
    # read training and testing data
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            features.append(item["histogram"])
            labels.append(item["rating"])

    #training
    training_Features = features[0:trainingSamples]
    training_Labels = labels[0:trainingSamples]
    # testing
    testing_Features = features[trainingSamples:trainingSamples + testingSamples]
    testing_Labels = labels[trainingSamples:trainingSamples + testingSamples]


inputfile = "data/output/histogram.json"
print("Preparing data ...")
Data_Preparation(inputfile)
print("Finished preparing data ...\n")

X_train = training_Features
y_train = training_Labels
y_train_new = []
np.zeros((3,3)).ravel()
for i in range(len(y_train)):
    temp = np.zeros((1,5)).ravel()
    temp[y_train[i]-1] = 1
    y_train_new.append(temp)

y_train_new = y_train_new.toarray()
print type(y_train_new)
  
X_test = testing_Features 
y_test = testing_Labels 
y_test_new = []
for i in range(len(y_test)):
    temp = [0, 0, 0, 0, 0]
    temp[y_train[i]-1] = 1
    y_test_new.append(temp)


# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_train, y_train_new).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 5
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_new[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_new.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot ROC curve
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
