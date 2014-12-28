#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import naive_bayes
import Scikit_Classification as sc


features = []
labels = []
   
def main():

    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    
    inputfile = "data/output/histogram_allFeatures.json"

    selectedFeatures = "all"
    
    trainingSamples = 800
    testingSamples = 200
    features, labels = sc.Data_Preparation(inputfile, selectedFeatures)

    Scikit_NaiveBayes_Model = naive_bayes.BernoulliNB()
    
    numberOfSamples = trainingSamples + testingSamples
    
    accuracy, testing_Labels, predict_Labels =  sc.Classification(Scikit_NaiveBayes_Model, features, labels, trainingSamples, testingSamples)
    sc.Result_Evaluation('data/evaluation_result/evaluation_NB.txt', accuracy, testing_Labels, predict_Labels)

    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





