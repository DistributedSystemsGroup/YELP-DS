#!/usr/bin/env python
# encoding: utf-8

from time import gmtime, strftime

from sklearn import cross_validation
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error as mse
import numpy as np

import io, os
import json

def Result_Evaluation (outputpath, accuracy, testing_Labels, predict_Labels):
    acc_rate = [0, 0, 0, 0, 0]
    testingSamples = len(testing_Labels)
    if os.path.isfile(outputpath):
        os.remove(outputpath)
    with io.open(outputpath, 'a', encoding='utf-8') as output_file:
        for i in xrange(0, testingSamples):
            rounded_result = int(round(predict_Labels[i]))
            if rounded_result == testing_Labels[i]:
                acc_rate[0] += 1
                result_item = str(i) + ": " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> spot on!\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=1:
                acc_rate[1] += 1
                result_item = str(i) + ": " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 1 star\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=2:
                acc_rate[2] += 1
                result_item = str(i) + ": " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 2 star\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=3:
                acc_rate[3] += 1
                result_item = str(i) + ": " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 3 star\n"
                output_file.write(unicode(result_item))
            else:
                acc_rate[4] += 1
                result_item = str(i) + ": " + str(predict_Labels[i]) + " - " + str(testing_Labels[i]) + " --> off by 4 star\n"
                output_file.write(unicode(result_item))

        #output_file.write(unicode(additional_info))
        finalResult = " #spot on: " + str(acc_rate[0]) + '\n' + " #off by 1 star: " + str(acc_rate[1]) + '\n' + " #off by 2 star: " + str(acc_rate[2]) + '\n' + " #off by 3 star: " + str(acc_rate[3]) + '\n' + " #off by 4 star: " + str(acc_rate[4]) + '\n'
        output_file.write(unicode(finalResult))

        finalResultPercentage = " #spot on: " + str(acc_rate[0]*1.0/testingSamples) + '\n' + " #off by 1 star: " + str(acc_rate[1]*1.0/testingSamples) + '\n' + " #off by 2 star: " + str(acc_rate[2]*1.0/testingSamples) + '\n' + " #off by 3 star: " + str(acc_rate[3]*1.0/testingSamples) + '\n' + " #off by 4 star: " + str(acc_rate[4]*1.0/testingSamples) + '\n'
        output_file.write(unicode(finalResultPercentage))
        print(" #Right: " + str(acc_rate[0]*1.0/testingSamples) + '\n')
        print(" #Wrong: " + str((acc_rate[2]+acc_rate[3]+acc_rate[4])*1.0/testingSamples) + '\n')
        r2Score = r2_score(testing_Labels, predict_Labels)
        print(" #R2 score: " + str(r2Score))
        print (" #sqrt(mse): {:f}".format(np.sqrt(mse(testing_Labels, predict_Labels))))
        print("Look at the evaluation_file for details!")

def Data_Preparation(filename, selectedFeatures):
    features = []
    labels = []
    with open(filename) as data_file:
        data = json.load(data_file)
        if selectedFeatures == "all":
            for item in data:            
                features.append(item["histogram"])                
                labels.append(item["rating"])
        else:
            for item in data:            
                features.append(list(item["histogram"][i] for i in selectedFeatures))
                labels.append(item["rating"])
       
    return features, labels

def Data_Preparation_TwoDataset(training_Filename, testing_Filename, selectedFeatures):

    training_Features = []
    training_Labels = []
    testing_Features = []
    testing_Labels = []

    trainingSamples = 0
    testingSamples = 0   


    train_features = []
    test_features = []    
    train_labels = []
    test_labels = []

    #training
    with open(training_Filename) as data_file:
        data = json.load(data_file)
        if selectedFeatures == "all":
            for item in data:            
                train_features.append(item["histogram"])                
                train_labels.append(item["rating"])
        else:
            for item in data:            
                train_features.append(list(item["histogram"][i] for i in selectedFeatures))
                train_labels.append(item["rating"])        
    
    training_Features = train_features    
    training_Labels = train_labels 

    # testing
    with open(testing_Filename) as data_file:
        data = json.load(data_file)
        if selectedFeatures == "all":
            for item in data:            
                test_features.append(item["histogram"])                
                test_labels.append(item["rating"])
        else:
            for item in data:            
                test_features.append(list(item["histogram"][i] for i in selectedFeatures))
                test_labels.append(item["rating"])      

    testing_Features = test_features    
    testing_Labels = test_labels
    #testing_Samples = len(test_features)

    return training_Features, training_Labels, testing_Features, testing_Labels


def Classification(model, features, labels, trainingSamples, testingSamples):

    training_Features = features[0:trainingSamples]
    training_Labels = labels[0:trainingSamples]
    testing_Features = features[trainingSamples:trainingSamples + testingSamples]
    testing_Labels = labels[trainingSamples:trainingSamples + testingSamples]

    print("Training ..")
    model.fit(training_Features, training_Labels)
    #print("Feature importances: ")
    #print(model.feature_importances_)
    print("Testing ..")
    predict_Labels = model.predict(testing_Features)
    accuracy = model.score(testing_Features, testing_Labels)

    return accuracy, testing_Labels, predict_Labels

def Classification_Bagging(model, features, labels, trainingSamples, testingSamples):

    training_Features = features[0:trainingSamples]
    training_Labels = labels[0:trainingSamples]
    testing_Features = features[trainingSamples:trainingSamples + testingSamples]
    testing_Labels = labels[trainingSamples:trainingSamples + testingSamples]
    
    bagging = BaggingClassifier(model, max_samples=0.5, max_features=1.0, n_estimators=10)

    print("Training ..")
    bagging.fit(training_Features, training_Labels)

    print("Testing ..")
    predict_Labels = bagging.predict(testing_Features)
    accuracy = bagging.score(testing_Features, testing_Labels)

    return accuracy, testing_Labels, predict_Labels

def Classification_Boosting(model, features, labels, trainingSamples, testingSamples):

    training_Features = features[0:trainingSamples]
    training_Labels = labels[0:trainingSamples]
    testing_Features = features[trainingSamples:trainingSamples + testingSamples]
    testing_Labels = labels[trainingSamples:trainingSamples + testingSamples]
    
    boosting = ensemble.AdaBoostClassifier(model, n_estimators=60, learning_rate=1)

    print("Training ..")
    boosting.fit(training_Features, training_Labels)

    print("Testing ..")
    predict_Labels = boosting.predict(testing_Features)
    accuracy = boosting.score(testing_Features, testing_Labels)

    return accuracy, testing_Labels, predict_Labels

def Classification_Stacking_Label(clfs, clfs_features, clfs_labels, stacking_model, trainingSamples, testingSamples):

    blending_training_Features = []
    blending_testing_Features = []

    training_Labels = clfs_labels[0:trainingSamples]
    testing_Labels = clfs_labels[trainingSamples:trainingSamples + testingSamples]

    numberOfClassifiers = len(clfs)

    training_Features = [0 for u in xrange(0, numberOfClassifiers)]
    testing_Features = [0 for u in xrange(0, numberOfClassifiers)]    
    clfs_Predict_Labels_Training = [0 for u in xrange(0, numberOfClassifiers)]
    clfs_Predict_Labels_Testing = [0 for u in xrange(0, numberOfClassifiers)]
    


    for i in xrange(0, numberOfClassifiers):
        training_Features[i] = clfs_features[i][0:trainingSamples]
        clfs[i].fit(training_Features[i], training_Labels)
        clfs_Predict_Labels_Training[i] = clfs[i].predict(training_Features[i])
        testing_Features[i] = clfs_features[i][trainingSamples:trainingSamples + testingSamples]
        clfs_Predict_Labels_Testing[i] = clfs[i].predict(testing_Features[i])

    
    for i in xrange(0, trainingSamples):        
        blending_item = [clfs_Predict_Labels_Training[u][i] for u in xrange(0, numberOfClassifiers)]        
        blending_training_Features.append(blending_item)

    stacking_model.fit(blending_training_Features, training_Labels)

    for i in xrange(0, testingSamples):        
        blending_item = [clfs_Predict_Labels_Testing[u][i] for u in xrange(0, numberOfClassifiers)]
        blending_testing_Features.append(blending_item)

    predict_Labels = stacking_model.predict(blending_testing_Features)
    accuracy = stacking_model.score(blending_testing_Features, testing_Labels)

    return accuracy, testing_Labels, predict_Labels


def Classification_Stacking_Proba(clfs, clfs_features, clfs_labels, stacking_model, trainingSamples, testingSamples):

    blending_training_Features = []
    blending_testing_Features = []

    training_Labels = clfs_labels[0:trainingSamples]
    testing_Labels = clfs_labels[trainingSamples:trainingSamples + testingSamples]

    numberOfClassifiers = len(clfs)

    training_Features = [0 for u in xrange(0, numberOfClassifiers)]
    testing_Features = [0 for u in xrange(0, numberOfClassifiers)]    
    clfs_Predict_Labels_Training = [0 for u in xrange(0, numberOfClassifiers)]
    clfs_Predict_Labels_Testing = [0 for u in xrange(0, numberOfClassifiers)]
    


    for i in xrange(0, numberOfClassifiers):
        training_Features[i] = clfs_features[i][0:trainingSamples]
        clfs[i].fit(training_Features[i], training_Labels)
        clfs_Predict_Labels_Training[i] = clfs[i].predict_proba(training_Features[i])
        testing_Features[i] = clfs_features[i][trainingSamples:trainingSamples + testingSamples]
        clfs_Predict_Labels_Testing[i] = clfs[i].predict_proba(testing_Features[i])

    
    for i in xrange(0, trainingSamples):
        blending_item = [clfs_Predict_Labels_Training[u][i][j] for u in xrange(0, numberOfClassifiers) for j in xrange(0, 5)]
        #print(blending_item)
        blending_training_Features.append(blending_item)

    stacking_model.fit(blending_training_Features, training_Labels)

    for i in xrange(0, testingSamples):        
        blending_item = [clfs_Predict_Labels_Testing[u][i][j] for u in xrange(0, numberOfClassifiers) for j in xrange(0, 5)]
        blending_testing_Features.append(blending_item)

    predict_Labels = stacking_model.predict(blending_testing_Features)
    accuracy = stacking_model.score(blending_testing_Features, testing_Labels)

    return accuracy, testing_Labels, predict_Labels

def Classification_Blending(model1, features1, labels1, model2, features2, labels2, trainingSamples, testingSamples):
    Scikit_LogisticRegression_Model = LogisticRegression()

    blending_training_Features = []
    blending_testing_Features = []

    training_Features1 = features1[0:trainingSamples]
    training_Labels1 = labels1[0:trainingSamples]
    model1.fit(training_Features1, training_Labels1)
    model1_Predict_Labels = model1.predict(training_Features1)

    testing_Features1 = features1[trainingSamples:trainingSamples + testingSamples]
    testing_Labels1 = labels1[trainingSamples:trainingSamples + testingSamples]

    training_Features2 = features2[0:trainingSamples]
    training_Labels2 = labels2[0:trainingSamples]
    model2.fit(training_Features2, training_Labels2)
    model2_Predict_Labels = model2.predict(training_Features2)


    testing_Features2 = features2[trainingSamples:trainingSamples + testingSamples]
    testing_Labels2 = labels2[trainingSamples:trainingSamples + testingSamples]

    for i in xrange(0, trainingSamples):        
        blending_item = [
                         model1_Predict_Labels[i],
                         model2_Predict_Labels[i]
                        ]
        blending_training_Features.append(blending_item)

    Scikit_LogisticRegression_Model.fit(blending_training_Features, training_Labels1)

    model1_Predict_Labels = model1.predict(testing_Features1)
    model2_Predict_Labels = model2.predict(testing_Features2)


    for i in xrange(0, testingSamples):        
        blending_item = [
                         model1_Predict_Labels[i],
                         model2_Predict_Labels[i]
                        ]
        blending_testing_Features.append(blending_item)

    predict_Labels = Scikit_LogisticRegression_Model.predict(blending_testing_Features)
    accuracy = Scikit_LogisticRegression_Model.score(blending_testing_Features, testing_Labels1)

    return accuracy, testing_Labels1, predict_Labels

def Classification_CrossValidation(model, features, labels, numberOfSamples, n_folds):

    kf = cross_validation.KFold(numberOfSamples, n_folds=n_folds, shuffle=False, random_state=None)
    X = np.array(features)
    y = np.array(labels)
    testingSamples = numberOfSamples/n_folds
    prediction_Labels = [[0 for u in xrange(0,n_folds)] for v in xrange(0,testingSamples)]
    
    k = 0
    accList = []
    for train_index, test_index in kf:        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("Training ..")
        model.fit(X_train, y_train)
        print("Testing ..")
        predict_Labels = model.predict(X_test)
        acc = model.score(X_test, y_test)
        print("accuracy " + str(k+1) + ": " + str(acc))
        accList.append(acc)
        for i in xrange(0, len(X_test)):
            prediction_Labels[i].__setitem__(k, predict_Labels[i])
        k += 1
    print("Avg accuracy: ")
    print(reduce(lambda x, y: x + y, accList) / len(accList))
    predicted_Labels = []
    for i, item in enumerate(prediction_Labels):
        predicted_Labels.append(max(set(prediction_Labels[i]), key=prediction_Labels[i].count))
    print(predicted_Labels)
    accuracy = model.score(X_test, predicted_Labels)
    print("Accuracy: " + str(accuracy))
    return accuracy, y_test, predicted_Labels
