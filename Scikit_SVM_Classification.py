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

    # train the model
    training_Features = features[0:800]
    training_Labels  = labels[0:800]
    Scikit_SVM_Model = svm.SVC()
    Scikit_SVM_Model.fit(training_Features, training_Labels)

    # testing
    testing_Features = features[800:1000]
    testing_Labels = labels[800:1000]

    predict_Labels = Scikit_SVM_Model.predict(testing_Features)
    print predict_Labels

def main():
    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    Scikit_SVM_Classification('output_split1000.json')
    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





