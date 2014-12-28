#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import ensemble
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

    Scikit_RandomForest_Model = ensemble.RandomForestClassifier(n_estimators=510, criterion='gini', max_depth=None,
                                                                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                                                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                                                                 min_density=None, compute_importances=None)

    
    numberOfSamples = trainingSamples + testingSamples
    #accuracy, testing_Labels, predict_Labels =  sc.Classification_CrossValidation(Scikit_RandomForest_Model, features, labels, numberOfSamples, 10)    
    accuracy, testing_Labels, predict_Labels =  sc.Classification(Scikit_RandomForest_Model, features, labels, trainingSamples, testingSamples)
    sc.Result_Evaluation('data/evaluation_result/evaluation_RF.txt', accuracy, testing_Labels, predict_Labels)

    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





