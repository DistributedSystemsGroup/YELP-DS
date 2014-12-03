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
from sklearn import cross_validation
from sklearn.metrics import r2_score

import json
import numpy as np

import io, os


features = []
labels = []

training_Features = []
training_Labels = []
testing_Features = []
testing_Labels = []

numberOfSamples = 1000
# the number of training and testing samples
trainingSamples = int(0.8 * numberOfSamples)
testingSamples = int(0.2 * numberOfSamples)

""" the result evaluation function, which defined Absolutely Right: predict label==real label;
                                                  Nearly Right: |predict label-real label|<=1;
                                                  Wrong: else cases;
"""

def Result_Evaluation (outputpath, testing_Labels, predict_Labels):
    acc_rate = [0, 0, 0]

    if os.path.isfile(outputpath):
        os.remove(outputpath)
    with io.open(outputpath, 'a', encoding='utf-8') as output_file:
        for i in xrange(0, testingSamples):
            rounded_result = int(round(predict_Labels[i]))
            if rounded_result == testing_Labels[i]:
                acc_rate[0] += 1
                result_item = str(i) + ": " + str(testing_Labels[i]) + " - " + str(predict_Labels[i]) + " - " + str(rounded_result) + " --> right\n"
                output_file.write(unicode(result_item))
            elif abs(rounded_result - testing_Labels[i])<=1:
                acc_rate[1] += 1
                result_item = str(i) + ": " + str(testing_Labels[i]) + " - " + str(predict_Labels[i]) + " - " + str(rounded_result) + " --> nearlyright\n"
                output_file.write(unicode(result_item))
            else:
                acc_rate[2] += 1
                result_item = str(i) + ": " + str(testing_Labels[i]) + " - " + str(predict_Labels[i]) + " - " + str(rounded_result) + " --> wrong\n"
                output_file.write(unicode(result_item))

        finalResult = "#AbsolutelyRight: " + str(acc_rate[0]) + " #NearlyRight: " + str(acc_rate[1]) + " #Wrong: " + str(acc_rate[2]) + '\n'
        output_file.write(unicode(finalResult))
        finalResultPercentage = "#AbsolutelyRight: " + str(acc_rate[0]*1.0/testingSamples) + " #NearlyRight: " + str(acc_rate[1]*1.0/testingSamples) + " #Wrong: " + str(acc_rate[2]*1.0/testingSamples) + '\n'
        output_file.write(unicode(finalResultPercentage))
        print(" #Wrong: " + str(acc_rate[2]*1.0/testingSamples))

def Data_Preparation(filename, selectedFeatures):

    global training_Features
    global training_Labels
    global testing_Features
    global testing_Labels


    global features
    global labels
    # read training and testing data
    with open(filename) as data_file:
        data = json.load(data_file)
        for item in data:
            features.append(list(item["histogram"][i] for i in selectedFeatures))
            #features.append(item["histogram"])
            labels.append(item["rating"])
        #print(features)
    #training
    training_Features = features[0:trainingSamples]
    training_Labels = labels[0:trainingSamples]
    # testing
    testing_Features = features[trainingSamples:trainingSamples + testingSamples]
    testing_Labels = labels[trainingSamples:trainingSamples + testingSamples]

def Scikit_SVM_Classification(evaluation_file, kernel_Index):
    print("Starting SVM Classification ...")
    """
    C: Penalty parameter C of the error term (default=1.0)
    kernel: It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
    degree: Degree of the polynomial kernel function ('poly')
    gamma: Kernel coefficient for 'rbf', 'poly' and 'sigm'.
    coef0: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    shrinking: Whether to use the shrinking heuristic
    probability: Whether to enable probability estimates
    tol: Tolerance for stopping criterion (default=1e-3)
    cache_size: Specify the size of the kernel cache (in MB)
    class_weight: {dict, 'auto'}, optional
    verbose: bool, default: False
    max_iter: Hard limit on iterations within solver, or -1 for no limit.
    random_state: int seed, RandomState instance, or None (default)

    LinearSVC: Similar to SVC with parameter kernel='linear', but implemented in terms of
     |  liblinear rather than libsvm, so it has more flexibility in the choice of
     |  penalties and loss functions and should scale better (to large numbers of
     |  samples).
     |  
     |  This class supports both dense and sparse input and the multiclass support
     |  is handled according to a one-vs-the-rest scheme.
    NuSVC: Nu-Support Vector Classification.
     |  
     |  Similar to SVC but uses a parameter to control the number of support
     |  vectors.
     |  
     |  The implementation is based on libsvm.
    NuSVR: Nu Support Vector Regression.
     |  
     |  Similar to NuSVC, for regression, uses a parameter nu to control
     |  the number of support vectors. However, unlike NuSVC, where nu
     |  replaces C, here nu replaces with the parameter epsilon of SVR.
     |  
     |  The implementations is a based on libsvm.
    OneClassSVM: Unsupervised Outliers Detection.
     |  
     |  Estimate the support of a high-dimensional distribution.
     |  
     |  The implementation is based on libsvm.
    SVC: C-Support Vector Classification.
     |  
     |  The implementations is a based on libsvm. The fit time complexity
     |  is more than quadratic with the number of samples which makes it hard
     |  to scale to dataset with more than a couple of 10000 samples.
     |  
     |  The multiclass support is handled according to a one-vs-one scheme.
     |  
     |  For details on the precise mathematical formulation of the provided
     |  kernel functions and how `gamma`, `coef0` and `degree` affect each,
     |  see the corresponding section in the narrative documentation:
     |  :ref:`svm_kernels`.
     |  
     |  .. The narrative documentation is available at http://scikit-learn.org
    SVR: epsilon-Support Vector Regression.
     |  
     |  The free parameters in the model are C and epsilon.
     |  
     |  The implementations is a based on libsvm.




    """
    Scikit_SVM_Model = svm.SVC(kernel='linear')
    #Scikit_SVM_Model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
    
    #Scikit_SVM_Model = svm.LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
    
    """
    if kernel_Index == 1:
        Scikit_SVM_Model = svm.SVC(kernel='linear')        
    elif kernel_Index == 2:
        Scikit_SVM_Model = svm.SVC(kernel='rbf')
    elif kernel_Index == 3:
        Scikit_SVM_Model = svm.SVC(kernel='poly', degree=3)
    """
    print("Training ..")
    Scikit_SVM_Model.fit(training_Features, training_Labels)
    print(training_Features)
    print("Testing ..")
    predict_Labels = Scikit_SVM_Model.predict(testing_Features)
    accuracy = Scikit_SVM_Model.score(testing_Features, testing_Labels)
    print "SVM_Classification: "
    print accuracy
    
    #the implementation of R_2 score measure 
    r2Score = r2_score(testing_Labels, predict_Labels)
    print("R2 Square score of SVM Classification: " + str(r2Score)) 

    Result_Evaluation (evaluation_file, testing_Labels, predict_Labels)

def Scikit_SVM_CrossValidation_Classification(evaluation_file, kernel_Index):
    print("Starting SVM Classification ...")
    if kernel_Index == 1:
        Scikit_SVM_Model = svm.SVC(kernel='linear')
    elif kernel_Index == 2:
        Scikit_SVM_Model = svm.SVC(kernel='rbf')
    elif kernel_Index == 3:
        Scikit_SVM_Model = svm.SVC(kernel='poly', degree=3)
    print("Training ..")
    
    kf = cross_validation.KFold(numberOfSamples, n_folds=5, shuffle=False, random_state=None)
    X = np.array(features)
    y = np.array(labels)
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        #print ("X_TRAIN:", X_train, "X_TEST:", X_test)
        y_train, y_test = y[train_index], y[test_index]
        #print ("y_TRAIN:", y_train, "y_TEST:", y_test)

        Scikit_SVM_Model.fit(X_train, y_train)
        print("Testing ..")
        predict_Labels = Scikit_SVM_Model.predict(X_test)
        accuracy = Scikit_SVM_Model.score(X_test, y_test)
        print "SVM_Classification: "
        print accuracy
        
        #the implementation of R_2 score measure 
        r2Score = r2_score(testing_Labels, predict_Labels)
        print("R2 Square score of SVM CrossValidation Classification: " + str(r2Score)) 
    

    #Result_Evaluation (evaluation_file, y_test, predict_Labels)

def Scikit_SVM_Regression(evaluation_file, kernel_Index):
    print("Starting SVM Regression ...")
    #(kernel can be 1, 2, 3)
    if kernel_Index == 1:
        Scikit_SVR_Model = svm.SVR(kernel='linear', C=1e3)
    elif kernel_Index == 2:
        Scikit_SVR_Model = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    elif kernel_Index == 3:
        Scikit_SVR_Model = svm.SVR(kernel='poly', C=1e3, degree=2)
    print("Training ..")
    Scikit_SVR_Model.fit(training_Features, training_Labels)
    print("Testing ..")
    predict_Labels = Scikit_SVR_Model.predict(testing_Features)

    accuracy = Scikit_SVR_Model.score(testing_Features, testing_Labels)
    print "SVMRegression_Classification: "
    print accuracy

    #the implementation of R_2 score measure
    #results = []
    #for i in xrange(0, len(testing_Labels)):
    #    results.append({'y_true': testing_Labels[i], 'y_pred': predict_Labels[i]})
    #sortedList = sorted(results, key=lambda k: k['y_true'])
    #print(sortedList)
    r2Score = r2_score(testing_Labels, predict_Labels)
    print("R2 Square score of SVM Regression Classification: " + str(r2Score)) 

    #Result_Evaluation (evaluation_file, testing_Labels, predict_Labels)

def Scikit_DecisionTree_Classification(evaluation_file):
    print("Starting Decision Tree Classification ...")
    Scikit_DecisionTree_Model = tree.DecisionTreeClassifier()
    print("Training ..")
    Scikit_DecisionTree_Model.fit(training_Features, training_Labels)
    print("Drawing tree ..")
    # Draw tree
    with open("data/output/others/tree.dot", 'w') as f:
        f = tree.export_graphviz(Scikit_DecisionTree_Model, out_file=f)

    print("Testing ..")
    predict_Labels = Scikit_DecisionTree_Model.predict(testing_Features)
    accuracy = Scikit_DecisionTree_Model.score(testing_Features, testing_Labels)
    print "DecisionTree_Classification: "
    print accuracy

    #the implementation of R_2 score measure 
    r2Score = r2_score(testing_Labels, predict_Labels)
    print("R2 Square score of Decision Classification: " + str(r2Score)) 

    Result_Evaluation (evaluation_file, testing_Labels, predict_Labels)

def Scikit_RandomForest_Classification(evaluation_file):
    print("Starting Random Forest Classification ...")
    #Scikit_RandomForest_Model = ensemble.RandomForestClassifier(n_estimators=500)
    """n_estimators: The number of trees in the forest.
       criterio: The function to measure the quality of a split
       max_depth: The maximum depth of the tree (integer, None, default=None)
       min_samples_split: The minimum number of samples required to split an internal node (integer, default=2)
       min_samples_leaf: The minimum number of samples in newly created leaves (integer, default=1)
       max_features: The function to measure the quality of a split (int, float, auto, sqrt, log2, None)
       bootstrap: Whether bootstrap samples are used when building trees (boolean, default=True)
       oob_score: Whether to use out-of-bag samples to estimate the generalization error (bool)
       n_jobs: The number of jobs to run in parallel for both `fit` and `predict`. If -1, then it is set to the number of cores (default=1)
       random_state: int, RandomState instance or None, optional (default=None)
       verbose: Controls the verbosity of the tree building process (default=0)
       min_density: not mentioned
       compute_importances: not mentioned
    """
    Scikit_RandomForest_Model = ensemble.RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None,
                                                                 min_samples_split=2, min_samples_leaf=1, max_features='auto',
                                                                 bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                                                 min_density=None, compute_importances=None)
    print("Training ..")
    Scikit_RandomForest_Model.fit(training_Features, training_Labels)
    print("Testing ..")
    predict_Labels = Scikit_RandomForest_Model.predict(testing_Features)
    accuracy = Scikit_RandomForest_Model.score(testing_Features, testing_Labels)

    print "RandomForest_Classification: "
    print accuracy

    #the implementation of R_2 score measure 
    r2Score = r2_score(testing_Labels, predict_Labels)
    print("R2 Square score of RandomForest Classification: " + str(r2Score)) 

    Result_Evaluation (evaluation_file, testing_Labels, predict_Labels)

   
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
    selectedFeatures = [6, 7]
    
    #print(features)
    Data_Preparation(inputfile, selectedFeatures)
    print("Finished preparing data ...")
    Scikit_SVM_Classification('data/evaluation_result/evaluation_SVM.txt', 1)
    #Scikit_SVM_CrossValidation_Classification('data/evaluation_result/evaluation_SVM_CV.txt', 1)
    #Scikit_SVM_Regression('data/evaluation_result/evaluation_SVMR.txt', 1)
    #Scikit_DecisionTree_Classification('data/evaluation_result/evaluation_DT.txt')
    #Scikit_RandomForest_Classification('data/evaluation_result/evaluation_RF.txt')



    endtime = strftime("%Y-%m-%d %H:%M:%S",gmtime())
    print(starttime)
    print(endtime)

if __name__  == "__main__":
    main()





