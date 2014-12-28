#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import ensemble
from sklearn import tree
import Scikit_Classification as sc


features = []
labels = []
   
def main():

    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    
    config = {}
    execfile("params.conf", config)
    inputfile = config["histogram_dataset"]    
    trainingSamples = config["trainingSamples"]
    testingSamples = config["testingSamples"]

    selectedFeatures = "all"

    features, labels = sc.Data_Preparation(inputfile, selectedFeatures)

    Scikit_AdaBoostDecisionTree_Model = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=7, max_features='sqrt'),
                                                      n_estimators=600, learning_rate=1)

    
    numberOfSamples = trainingSamples + testingSamples
    
    accuracy, testing_Labels, predict_Labels =  sc.Classification(Scikit_AdaBoostDecisionTree_Model, features, labels, trainingSamples, testingSamples)
    sc.Result_Evaluation('data/evaluation_result/evaluation_AdaDT.txt', accuracy, testing_Labels, predict_Labels)

    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





