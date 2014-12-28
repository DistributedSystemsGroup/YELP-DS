#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import ensemble, svm
import Scikit_Classification as sc


features = []
labels = []
   
def main():
    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    
    inputfile = "data/output/histogram_allFeatures.json"
    trainingSamples = 800
    testingSamples = 200
    numberOfSamples = trainingSamples + testingSamples

    rf_selectedFeatures = "all"
    svm_selectedFeatures = [20, 21, 22, 23, 24]

    
    rf_features, rf_labels = sc.Data_Preparation(inputfile, rf_selectedFeatures)
    svm_features, svm_labels = sc.Data_Preparation(inputfile, svm_selectedFeatures)

    Scikit_RandomForest_Model = ensemble.RandomForestClassifier(n_estimators=510, criterion='gini', max_depth=7,
                                                                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                                                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                                                                 min_density=None, compute_importances=None)

    Scikit_SVM_Model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

    
    accuracy, testing_Labels, predict_Labels =  sc.Classification_Blending(Scikit_RandomForest_Model, rf_features, rf_labels, Scikit_SVM_Model, svm_features, svm_labels, trainingSamples, testingSamples)
    
    sc.Result_Evaluation('data/evaluation_result/evaluation_Blending.txt', accuracy, testing_Labels, predict_Labels)


    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





