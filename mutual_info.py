#! /usr/bin/env python 

from sklearn import metrics
import json

def mutualScore(label1, label2):
    score = metrics.mutual_info_score(label1, label2)
    return score
    
def main():

   features = []
   labels = []

   with open("data/output/histogram.json") as dataFile:
       data = json.load(dataFile)
       for item in data:
           features.append(item["histogram"])
           labels.append(item["rating"])

   for i in range(len(features[0])):
       selected_features = []
       for row in features:
           selected_features.append(row[i])
       print("Feature " + str(i) + " score = " + str(mutualScore(selected_features, labels)))
   
       

if __name__ == "__main__":
    main()
 
