#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime

from sklearn import ensemble, svm, tree, naive_bayes
import Scikit_Classification as sc
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression



features = []
labels = []
   
def main():
    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    
    config = {}
    execfile("params.conf", config)
    inputfile = config["histogram_dataset"]    
    trainingSamples = config["trainingSamples"]
    testingSamples = config["testingSamples"]

    numberOfSamples = trainingSamples + testingSamples

    rf_selectedFeatures = "all"
    svm_selectedFeatures = [20, 21, 22, 23, 24]
    dt_selectedFeatures = "all"
    nb_selectedFeatures = "all"
        
    rf_features, rf_labels = sc.Data_Preparation(inputfile, rf_selectedFeatures)
    svm_features, svm_labels = sc.Data_Preparation(inputfile, svm_selectedFeatures)
    dt_features, dt_labels = sc.Data_Preparation(inputfile, dt_selectedFeatures)
    nb_features, nb_labels = sc.Data_Preparation(inputfile, nb_selectedFeatures)

    # Model declaration

    RandomForest_Classification = ensemble.RandomForestClassifier(n_estimators=510, criterion='gini', max_depth=7,
                                                                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                                                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                                                                 min_density=None, compute_importances=None)

    SVM_Classification = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

    DecisionTree_Classification = tree.DecisionTreeClassifier(max_depth=7, max_features='sqrt')
    AdaBoostDecisionTree_Classification = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=7, max_features='sqrt'),
                                                      n_estimators=600, learning_rate=1)

    NaiveBayes_Classification = naive_bayes.BernoulliNB()

    RandomForest_Regression = ensemble.RandomForestRegressor(n_estimators=510, criterion='mse', max_depth=7, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, min_density=None, compute_importances=None)

    SVM_Regression = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
  
    DecisionTree_Regression = tree.DecisionTreeRegressor(max_depth=7)

    AdaBoostDecisionTree_Regression = ensemble.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=7), n_estimators=500, learning_rate=1.0, loss='linear', random_state=None)

    Bayesian_Regression = BayesianRidge(compute_score=True)

    # Change stacking scenario here

    clfs = [RandomForest_Regression, SVM_Regression, AdaBoostDecisionTree_Regression, Bayesian_Regression]

    clfs_features = [rf_features, svm_features, dt_features, nb_features]
    clfs_labels = rf_labels

    stacking_model = LinearRegression()    #LogisticRegression()
    
    #accuracy, testing_Labels, predict_Labels = sc.Classification(AdaBoostDecisionTree_Regression, dt_features, dt_labels, trainingSamples, testingSamples)

    accuracy, testing_Labels, predict_Labels = sc.Classification_Stacking_Label(clfs, clfs_features, clfs_labels, stacking_model, trainingSamples, testingSamples)
    
    sc.Result_Evaluation('data/evaluation_result/evaluation_Blending.txt', accuracy, testing_Labels, predict_Labels)

    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





