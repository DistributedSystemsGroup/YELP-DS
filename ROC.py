import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

def drawROCCurve(features, output):
    # Import some data to play with
    X = []
    y = []
    # read training and testing data
    fileinput = "data/output/histogram_allFeatures.json"
    with open(fileinput) as data_file:
        data = json.load(data_file)
        for item in data:
            X.append(list(item["histogram"][i] for i in features))
            y.append(item["rating"])
    X = np.array(X)
    y = np.array(y)
    # Binarize the output
    y = label_binarize(y, classes=[1, 2, 3, 4, 5])
    print(y)
    n_classes = y.shape[1]
    
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    #X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)
    print type(y_test)
    
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    # Save multiple plots to one pdf file
    pp = PdfPages(output)
    # Plot of a ROC curve for a specific class
    """nClass = 2
    plt.figure()
    plt.plot(fpr[nClass], tpr[nClass], label='ROC curve (area = %0.2f)' % roc_auc[nClass])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(pp, format='pdf')
    """
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    
    #plt.show()
    plt.savefig(pp, format='pdf')
    pp.close()


def main():
    features = []
    
    features1 = [0, 1, 2, 3, 4, 5, 8]
    features2 = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    features3 = [0, 1, 2, 3, 4, 5, 8, 14, 15, 16, 17, 18, 19]
    features4 = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    features5 = [6, 7, 8]
    features6 = [6, 7, 8, 9, 10, 11]
    features7 = [6, 7, 8, 14, 15, 16, 17, 18, 19]
    features8 = [6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    features9 = [12, 13, 8]
    features10 = [12, 13, 8, 9, 10, 11]
    features11 = [12, 13, 8, 14, 15, 16, 17, 18, 19]
    features12 = [12, 13, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    features13 = [20, 21, 22, 23, 24]
    features14 = [20, 21, 22, 23, 24, 9, 10, 11]
    features15 = [20, 21, 22, 23, 24, 14, 15, 16, 17, 18, 19]
    
    features.append(features1)
    features.append(features2)
    features.append(features3)
    features.append(features4)
    features.append(features5)
    features.append(features6)
    features.append(features7)
    features.append(features8)
    features.append(features9)
    features.append(features10)
    features.append(features11)
    features.append(features12)
    features.append(features13)
    features.append(features14)
    features.append(features15)

    print(features)
    for i in xrange(1, 16):
        print(i)
        output = 'report/refs/multipage' + str(i) + '.pdf'
        drawROCCurve(features[i-1], output)
    
if __name__ == "__main__":
    main()
