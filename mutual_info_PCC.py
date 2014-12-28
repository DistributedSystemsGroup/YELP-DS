#! /usr/bin/env python 

from sklearn import metrics
import json
from scipy.stats import pearsonr

#compute the mutual information between two matrixes
def mutualScore(label1, label2):
    score = metrics.mutual_info_score(label1, label2)
    return score

#compute the person correlation coefficient between two datasets,
#it measures the linear relationship, positive correlations imply 
#that as x increases, so does y.    
def personR(label1, label2):
    score = pearsonr(label1, label2)
    return score[0]

def main():

   features = []
   labels = []
   datafile = "data/output/histogram_allFeatures.json"
   with open(datafile) as dataFile:
       data = json.load(dataFile)
       for item in data:
           features.append(item["histogram"])
           labels.append(item["rating"])

   for i in range(len(features[0])):
       selected_features = []
       for row in features:
           selected_features.append(row[i])
       print("Feature " + str(i) + " mutual information score = " + str(mutualScore(selected_features, labels)))
       print("Feature " + str(i) + " person correlation coefficient score = " + str(personR(selected_features, labels)))
       

if __name__ == "__main__":
    main()
 
