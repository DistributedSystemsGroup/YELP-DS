#!/usr/bin/env python
# encoding: utf-8

"""
This code implemented review texts classication by using Support Vector Machine, Support Vector Regression, 
Decision Tree and Random Forest, the evaluation function has been implemented as well.
"""

from time import gmtime, strftime
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import cross_validation
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

import numpy as np
from random import randrange

import io, os
import json


training_Features = []
training_Labels = []
testing_Features = []
testing_Labels = []

svm_training_Features = []
svm_testing_Features = []

numberOfSamples = 1000
# the number of training and testing samples
trainingSamples = int(0.8 * numberOfSamples)
testingSamples = int(0.2 * numberOfSamples)

def Data_Preparation(filename, selectedFeatures):

    global training_Features
    global training_Labels
    global testing_Features
    global testing_Labels

    global svm_training_Features
    global svm_testing_Features


    features = []
    svm_features = []
    labels = []

    # read training and testing data
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            features.append(item["histogram"])
            svm_features.append(list(item["histogram"][i] for i in selectedFeatures))
            labels.append(item["rating"])
        #print(features)
    #training
    training_Features = features[0:trainingSamples]
    svm_training_Features = svm_features[0:trainingSamples]
    training_Labels = labels[0:trainingSamples]
    # testing
    testing_Features = features[trainingSamples:trainingSamples + testingSamples]
    svm_testing_Features = svm_features[trainingSamples:trainingSamples + testingSamples]
    testing_Labels = labels[trainingSamples:trainingSamples + testingSamples]

def Result_Evaluation (outputpath, testing_Labels, predict_Labels, blending_testing_Features):
    acc_rate = [0, 0, 0, 0, 0]

    #if os.path.isfile(outputpath):
        #os.remove(outputpath)
    with io.open(outputpath, 'a', encoding='utf-8') as output_file:
        for i in xrange(0, testingSamples):
            rounded_result = int(round(predict_Labels[i]))
            if rounded_result == testing_Labels[i]:
                acc_rate[0] += 1
                result_item = str(i) + ": " + str(blending_testing_Features[i]) + " - " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> spot on!\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=1:
                acc_rate[1] += 1
                result_item = str(i) + ": " + str(blending_testing_Features[i]) + " - " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 1 star\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=2:
                acc_rate[2] += 1
                result_item = str(i) + ": " + str(blending_testing_Features[i]) + " - " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 2 star\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=3:
                acc_rate[3] += 1
                result_item = str(i) + ": " + str(blending_testing_Features[i]) + " - " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 3 star\n"
                output_file.write(unicode(result_item))
            else:
                acc_rate[4] += 1
                result_item = str(i) + ": " + str(blending_testing_Features[i]) + " - " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> wrong\n"
                output_file.write(unicode(result_item))

        #output_file.write(unicode(additional_info))
        finalResult = "#spot on: " + str(acc_rate[0]) + '\n' + " #off by 1 star: " + str(acc_rate[1]) + '\n' + " #off by 2 star: " + str(acc_rate[2]) + '\n' + " #off by 3 star: " + str(acc_rate[3]) + '\n' + " #Wrong: " + str(acc_rate[4]) + '\n'
        output_file.write(unicode(finalResult))

        finalResultPercentage = "#spot on: " + str(acc_rate[0]*1.0/testingSamples) + '\n' + " #off by 1 star: " + str(acc_rate[1]*1.0/testingSamples) + '\n' + " #off by 2 star: " + str(acc_rate[2]*1.0/testingSamples) + '\n' + " #off by 3 star: " + str(acc_rate[3]*1.0/testingSamples) + '\n' + " #Wrong: " + str(acc_rate[4]*1.0/testingSamples) + '\n'
        output_file.write(unicode(finalResultPercentage))
        print(" #Wrong: " + str((acc_rate[2]+acc_rate[3]+acc_rate[4])*1.0/testingSamples))


def Scikit_Blending_Classification(evaluation_file):
    if os.path.isfile(evaluation_file):
        os.remove(evaluation_file)
    print("Starting Blending Classification ...")

    Scikit_AdaBoostDecisionTree_Model = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=7, max_features='sqrt'),
                                                      n_estimators=600, learning_rate=1)
    Scikit_RandomForest_Model = ensemble.RandomForestClassifier(n_estimators=510, criterion='gini', max_depth=7,
                                                                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                                                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0,
                                                                 min_density=None, compute_importances=None)
    #BaseWeightBoosting, BaseGradientBoosting
    Scikit_NaiveBayes_Model = naive_bayes.BernoulliNB()

    Scikit_SVM_Model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

    Scikit_BaggingSVM_Model = ensemble.AdaBoostClassifier(svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None), n_estimators=100, learning_rate=1.0)

    Scikit_LogisticRegression_Model = LogisticRegression()

    blending_training_Features = []
    blending_testing_Features = []


    print("Training ..")
    #Scikit_AdaBoostDecisionTree_Model.fit(training_Features, training_Labels)
    Scikit_RandomForest_Model.fit(training_Features, training_Labels)
    #Scikit_NaiveBayes_Model.fit(training_Features, training_Labels)
    Scikit_SVM_Model.fit(svm_training_Features, training_Labels)
    #Scikit_BaggingSVM_Model.fit(svm_training_Features, training_Labels)

    print("Testing ..")
    #predict_Labels = Scikit_NaiveBayes_Model.predict(testing_Features)
    #AdaBoostDecisionTree_Predict_Probas = Scikit_AdaBoostDecisionTree_Model.predict_proba(training_Features)
    RandomForest_Predict_Probas = Scikit_RandomForest_Model.predict_proba(training_Features)
    #NaiveBayes_Predict_Probas = Scikit_NaiveBayes_Model.predict_proba(training_Features)
    SVM_Predict_Probas = Scikit_SVM_Model.predict_proba(svm_training_Features)
    #BaggingSVM_Predict_Probas = Scikit_BaggingSVM_Model.predict_proba(svm_training_Features)

    #Predict labels
    #AdaBoostDecisionTree_Predict_Labels = Scikit_AdaBoostDecisionTree_Model.predict(training_Features)
    RandomForest_Predict_Labels = Scikit_RandomForest_Model.predict(training_Features)
    #NaiveBayes_Predict_Labels = Scikit_NaiveBayes_Model.predict(training_Features)
    SVM_Predict_Labels = Scikit_SVM_Model.predict(svm_training_Features)
    #BaggingSVM_Predict_Labels = Scikit_BaggingSVM_Model.predict(svm_training_Features)
    
    for i in xrange(0, trainingSamples):
        blending_item = [
                         #AdaBoostDecisionTree_Predict_Labels[i],
                         RandomForest_Predict_Labels[i],
                         #NaiveBayes_Predict_Probas[i][0],
                         #NaiveBayes_Predict_Probas[i][1],
                         #NaiveBayes_Predict_Probas[i][2],
                         #NaiveBayes_Predict_Probas[i][3],
                         #NaiveBayes_Predict_Probas[i][4],
                         SVM_Predict_Labels[i]
                        ]
        """blending_item = np.concatenate((AdaBoostDecisionTree_Predict_Probas[i], RandomForest_Predict_Probas[i], NaiveBayes_Predict_Probas[i], SVM_Predict_Probas[i], BaggingSVM_Predict_Probas[i]), axis=0)
        blending_training_Features.append(list(blending_item))"""
        blending_training_Features.append(blending_item)
        """for j, item in enumerate(blending_item):
            blending_training_Features[j].append(item)
        blending_training_Features[i].append(AdaBoostDecisionTree_Predict_Probas[i])
        blending_training_Features[i].append(RandomForest_Predict_Probas[i])
        blending_training_Features[i].append(NaiveBayes_Predict_Probas[i])"""

    #print(blending_training_Features[1])
    #print(training_Features[1])
    Scikit_LogisticRegression_Model.fit(blending_training_Features, training_Labels)

    #AdaBoostDecisionTree_Predict_Probas = Scikit_AdaBoostDecisionTree_Model.predict_proba(testing_Features)
    RandomForest_Predict_Probas = Scikit_RandomForest_Model.predict_proba(testing_Features)
    #NaiveBayes_Predict_Probas = Scikit_NaiveBayes_Model.predict_proba(testing_Features)
    SVM_Predict_Probas = Scikit_SVM_Model.predict_proba(svm_testing_Features)
    #BaggingSVM_Predict_Probas = Scikit_BaggingSVM_Model.predict_proba(svm_testing_Features)

    #Predict labels
    """
    AdaBoostDecisionTree_Predict_Labels = Scikit_AdaBoostDecisionTree_Model.predict(testing_Features)
    AdaBoostDecisionTree_Accuracy = Scikit_AdaBoostDecisionTree_Model.score(testing_Features, testing_Labels)
    print ("AdaBoostDecisionTree_Accuracy: ")
    print (AdaBoostDecisionTree_Accuracy)"""

    RandomForest_Predict_Labels = Scikit_RandomForest_Model.predict(testing_Features)
    RandomForest_Accuracy = Scikit_RandomForest_Model.score(testing_Features, testing_Labels)
    print ("RandomForest_Accuracy: ")
    print (RandomForest_Accuracy)

    """
    NaiveBayes_Predict_Labels = Scikit_NaiveBayes_Model.predict(testing_Features)
    NaiveBayes_Accuracy = Scikit_NaiveBayes_Model.score(testing_Features, testing_Labels)
    print ("NaiveBayes_Accuracy: ")
    print (NaiveBayes_Accuracy)"""

    SVM_Predict_Labels = Scikit_SVM_Model.predict(svm_testing_Features)
    SVM_Accuracy = Scikit_SVM_Model.score(svm_testing_Features, testing_Labels)
    print ("SVM_Accuracy: ")
    print (SVM_Accuracy)
     
    """BaggingSVM_Predict_Labels = Scikit_BaggingSVM_Model.predict(svm_testing_Features)
    BaggingSVM_Accuracy = Scikit_BaggingSVM_Model.score(svm_testing_Features, testing_Labels)
    print ("BaggingSVM_Accuracy: ")
    print (BaggingSVM_Accuracy)"""

    for i in xrange(0, testingSamples):
        blending_item = [
                         #AdaBoostDecisionTree_Predict_Labels[i],
                         RandomForest_Predict_Labels[i],
                         #NaiveBayes_Predict_Probas[i][0],
                         #NaiveBayes_Predict_Probas[i][1],
                         #NaiveBayes_Predict_Probas[i][2],
                         #NaiveBayes_Predict_Probas[i][3],
                         #NaiveBayes_Predict_Probas[i][4],
                         SVM_Predict_Labels[i]
                        ]
        """blending_item = np.concatenate((AdaBoostDecisionTree_Predict_Probas[i], RandomForest_Predict_Probas[i], NaiveBayes_Predict_Probas[i], SVM_Predict_Probas[i], BaggingSVM_Predict_Probas[i]), axis=0)
        blending_testing_Features.append(list(blending_item))"""
        blending_testing_Features.append(blending_item)
        """for j, item in enumerate(blending_item):
            blending_training_Features[j].append(item)
        blending_testing_Features[i].append(AdaBoostDecisionTree_Predict_Probas[i])
        blending_testing_Features[i].append(RandomForest_Predict_Probas[i])
        blending_testing_Features[i].append(NaiveBayes_Predict_Probas[i])"""

    #print(blending_testing_Features[1])    

    predict_Labels = Scikit_LogisticRegression_Model.predict(blending_testing_Features)
    accuracy = Scikit_LogisticRegression_Model.score(blending_testing_Features, testing_Labels) 

    print(accuracy)
    Result_Evaluation (evaluation_file, testing_Labels, predict_Labels, blending_testing_Features)
    
def main():
    starttime = strftime("%Y-%m-%d %H:%M:%S",gmtime())

    inputfile = "bow/data/output/histogram_allFeatures.json"

    #selectedFeatures = [0, 1, 2, 3, 4, 5, 8]
    #selectedFeatures = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    #selectedFeatures = [0, 1, 2, 3, 4, 5, 8, 14, 15, 16, 17, 18, 19]
    #selectedFeatures = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    #selectedFeatures = [6, 7, 8]
    #selectedFeatures = [6, 7, 8, 9, 10, 11]
    #selectedFeatures = [6, 7, 8, 14, 15, 16, 17, 18, 19]
    #selectedFeatures = [6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    #selectedFeatures = [12, 13, 8]
    #selectedFeatures = [12, 13, 8, 9, 10, 11]
    #selectedFeatures = [12, 13, 8, 14, 15, 16, 17, 18, 19]
    #selectedFeatures = [12, 13, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    #selectedFeatures = [20, 21, 22, 23, 24]
    #selectedFeatures = [20, 21, 22, 23, 24, 9, 10, 11]
    #selectedFeatures = [20, 21, 22, 23, 24, 14, 15, 16, 17, 18, 19]

    #selectedFeatures = [8, 12, 13, 14, 17, 20, 21, 22, 23, 24]
    selectedFeatures = [20, 21, 22, 23, 24]
    #selectedFeatures = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    
    #print(features)
    Data_Preparation(inputfile, selectedFeatures)
    print("Finished preparing data ...")
    
    Scikit_Blending_Classification('data/evaluation_result/evaluation_Blending.txt')



    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





