#!/usr/bin/env python
# encoding: utf-8

from time import gmtime, strftime
from sklearn import svm
import json

features = []
labels = []

def Scikit_SVM_Classification(filename):
    # read training and testing data
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            features.append(item["histogram"])
            labels.append(item["rating"])

    # train the model using linear kernel, RBF kernel and polynomial
    # kernel respecively
    training_Features = features[0:800]
    training_Labels  = labels[0:800]
    kernel_Index = 2 #(kernel can be 1, 2, 3)
    if kernel_Index == 1:
        Scikit_SVM_Model = svm.SVC(kernel='linear')
    elif kernel_Index == 2:
        Scikit_SVM_Model = svm.SVC(kernel='rbf')
    elif kernel_Index == 3:
        Scikit_SVM_Model = svm.SVC(kernel='poly', degree=3)
    Scikit_SVM_Model.fit(training_Features, training_Labels)

    # testing
    testing_Features = features[800:1000]
    testing_Labels = labels[800:1000]

    predict_Labels = Scikit_SVM_Model.predict(testing_Features)
    accurancy = Scikit_SVM_Model.score(testing_Features, testing_Labels)
    print accurancy

def Scikit_SVM_Regression(filename):
    # read training data and testing data
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            features.append(item['histogram'])
            features.append(item['rating'])

    # fit regression model
    training_Features = features[0:800]
    training_Labels  = labels[0:800]
    kernel_Index = 2 #(kernel can be 1, 2, 3)
    if kernel_Index == 1:
        Scikit_SVR_Model = svm.SVR(kernel='linear', C=1e3)
    elif kernel_Index == 2:
        Scikit_SVR_Model = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    elif kernel_Index == 3:
        Scikit_SVR_Model = svm.SVR(kernel='poly', C=1e3, degree=2)
    Scikit_SVR_Model.fit(training_Features, training_Labels)

    # testing
    testing_Features = features[800:1000]
    testing_Labels = labels[800:1000]

    predict_Labels = Scikit_SVR_Model.predict(testing_Features)
    print predict_Labels
    print testing_Labels-predict_Labels


def main():
    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    Scikit_SVM_Classification('output_split1000.json')
    Scikit_SVM_Regression('output_split1000.json')
    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





