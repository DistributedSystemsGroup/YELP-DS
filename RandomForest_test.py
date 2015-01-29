#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import ensemble
import Scikit_Classification as sc
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import GradientBoostingClassifier


features = []
labels = []
   
def main():

    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    config = {}
    execfile("params.conf", config)
    inputfile = config["histogram_dataset"]    
    trainingSamples = config["trainingSamples"]
    testingSamples = config["testingSamples"]

    #selectedFeatures = "all"
    selectedFeatures = [20, 21, 22, 23, 24]
    features, labels = sc.Data_Preparation(inputfile, selectedFeatures)

    Scikit_RandomForest_Model = ensemble.RandomForestClassifier(n_estimators=510, criterion='gini', max_depth=None,
                                                                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                                                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                                                                 min_density=None, compute_importances=None)
    Scikit_SVM_Model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    
    Scikit_GradientBoostingClassifier_Model = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=1, random_state=0)

    # ANOVA SVM-C
    anova_filter = SelectKBest(f_regression, k=5)

    clf = Pipeline([
        #('feature_selection', svm.LinearSVC(C=0.01, penalty="l1", dual=False)),
        ('anova', anova_filter),
        ('classification', Scikit_SVM_Model)
    ])
    numberOfSamples = trainingSamples + testingSamples
    #accuracy, testing_Labels, predict_Labels =  sc.Classification_CrossValidation(Scikit_RandomForest_Model, features, labels, numberOfSamples, 10)    
    accuracy, testing_Labels, predict_Labels =  sc.Classification(Scikit_GradientBoostingClassifier_Model, features, labels, trainingSamples, testingSamples)
    sc.Result_Evaluation('data/evaluation_result/evaluation_RF.txt', accuracy, testing_Labels, predict_Labels)

    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





