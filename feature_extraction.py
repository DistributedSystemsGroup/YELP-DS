from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm

from sklearn.preprocessing import normalize, scale
from sklearn import preprocessing
import numpy as np
import re, io, json
from string import punctuation

from time import gmtime, strftime
from datetime import datetime

def words(text):
    """An iterator over tokens (words) in text. Replace this with a
    stemmer or other smarter logic.
    """

    for word in text.split():
        # normalize words by lowercasing and dropping non-alpha characters
        normed = re.sub('[^a-z]', '', word.lower())
        if normed:
            yield normed

def buildingHistogram6Bins(keyword1, keyword2, result):
    if keyword1 == 'strongsubj' and keyword2 == 'positive':
        result[0]+=1
    elif keyword1 == 'weaksubj' and keyword2 == 'positive':
        result[1]+=1
    elif keyword1 == 'strongsubj' and keyword2 == 'neutral':
        result[2]+=1
    elif keyword1 == 'weaksubj' and keyword2 == 'neutral':
        result[3]+=1
    elif keyword1 == 'strongsubj' and keyword2 == 'negative':
        result[4]+=1
    elif keyword1 == 'weaksubj' and keyword2 == 'negative':
        result[5]+=1
    elif keyword1 == 'weaksubj' and keyword2 == 'both':
        result[1]+=1
        result[5]+=1
    elif keyword1 == 'strongsubj' and keyword2 == 'both':
        result[0]+=1
        result[4]+=1

def buildingHistogram3Bins(keyword, result):
    if keyword == 'positive':
        result[6]+=1
    elif keyword == 'negative':
        result[7]+=1
    elif keyword == 'negation':
        result[8]+=1

def buildingHistogram2Bins(ps, ns, result):
    result[12]+=float(ps)
    result[13]+=float(ns)

def buildingHistogramDay(date, result):
    date_object = datetime.strptime(date, '%Y-%m-%d')
    day = date_object.strftime("%A")
    if day == 'Sunday' or day == 'Saturday':
        result[10] = 1
    else:
        result[10] = 0

def buildingHistogramVote(votes, result):
    v_result = votes["funny"] + votes["useful"] + votes["cool"]
    result[11] = v_result
    
def buildingHistogramLength(length, result):
    result [9] = length

starttime = strftime("%Y-%m-%d %H:%M:%S", gmtime())

filenames = ['data/input/bow/1StarsSamples.json', 'data/input/bow/2StarsSamples.json', 'data/input/bow/3StarsSamples.json', 'data/input/bow/4StarsSamples.json', 'data/input/bow/5StarsSamples.json']

vectorizer = CountVectorizer(input='filename', ngram_range=(1,3), stop_words='english', strip_accents='unicode', token_pattern=ur'\b\w+\b')

dtm = vectorizer.fit_transform(filenames).toarray()
dtm = scale(dtm)
vocab = np.array(vectorizer.get_feature_names())

_vectorizer = CountVectorizer(input='content', ngram_range=(1,3), stop_words='english', strip_accents='unicode', token_pattern=ur'\b\w+\b')
analyze = _vectorizer.build_analyzer()

#Edit input and output file here
filename = '400000trainingSamples.json'
inputFile = 'data/input/' + filename
outputFile = 'data/output/histogram' + filename

with open(inputFile) as fileobject, open("data/dictionary/mydictionary_2bins.json") as dicObject2Bins, open("data/dictionary/mydictionary_3bins.json") as dicObject3Bins, open("data/dictionary/mydictionary_6bins.json") as dicObject6Bins:
    dicData2Bins = json.load(dicObject2Bins)
    dicData3Bins = json.load(dicObject3Bins)
    dicData6Bins = json.load(dicObject6Bins)
    listOfHistogramAndRating = []

    for i, line in enumerate(fileobject):

        print(i)

        if line == '\n':
            break
        data = json.loads(line)
        #Descripton [0-spos, 1-wpos, 2-sneu, 3-wneu, 4-sneg, 5-wneg, 6-pos, 7-neg, 8-negation, 9-len, 10-day, 11-vote, 12-ps, 13-ns]
        result =    [0     , 0     , 0     , 0     , 0     , 0     , 0    , 0    , 0         , 0    , 0     , 0      , 0.0  , 0.0  ]
        length = 0
        text = data["text"]
        for word in words(text):
            length += 1

            for item in dicData2Bins:
                if word == item["word"]:
                    buildingHistogram2Bins(item["pos"], item["neg"], result)
            for item in dicData3Bins:
                if word == item["word"]:
                    buildingHistogram3Bins(item["priorpolarity"], result)
            for item in dicData6Bins:
                if word == item["word"]:
                    buildingHistogram6Bins(item["type"], item["priorpolarity"], result)
        
        buildingHistogramDay(data["date"], result)

        buildingHistogramVote(data["votes"], result)
                
        buildingHistogramLength(length, result)

        character_count = len(text)
        uppercase_count = sum(1 for c in text if c.isupper())
        lowercase_count = sum(1 for c in text if c.islower())
        punctuation_count = sum(1 for c in text if c in punctuation)
        alphabetic_count = sum(1 for c in text if c.isalpha())
        numeric_count = sum(1 for c in text if c.isnumeric())
        
       
        #Fearures lesection        
        final_result = []
               

        final_result.append(result[0])
        final_result.append(result[1])
        final_result.append(result[2])
        final_result.append(result[3])
        final_result.append(result[4])
        final_result.append(result[5])
        final_result.append(result[6])
        final_result.append(result[7])
        final_result.append(result[8])
        final_result.append(result[9])
        final_result.append(result[10])
        final_result.append(result[11])
        final_result.append(result[12])
        final_result.append(result[13])

        final_result.append(character_count) #14
        final_result.append(uppercase_count) #15
        final_result.append(lowercase_count) #16
        final_result.append(punctuation_count) #17
        final_result.append(alphabetic_count) #18
        final_result.append(numeric_count) #19

        analyzed = analyze(text)
        #Descripton [20-1star, 21-2star, 22-3star, 23-4star, 24-5star]
        resultBOW = [0       , 0       , 0       , 0       , 0       ]
        
        for item in analyzed:
            feature_index = vectorizer.vocabulary_.get(item)
            if feature_index:
                item_histogram = dtm[:, feature_index]
                
                resultBOW = [x + y for x, y in zip(resultBOW, item_histogram)]
        
        for item in resultBOW:
            if sum(resultBOW) != 0:
                item = item*1.0/sum(resultBOW)
            final_result.append(item)

        
        listOfHistogramAndRating.append({'rating': data["stars"], 'histogram': final_result})
    with io.open(outputFile, 'w', encoding='utf-8') as outfile:
        outfile.write(unicode(json.dumps(listOfHistogramAndRating, ensure_ascii=False)))


print "FINISHED"

endtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())       

mailmessage = 'This is a notification email to show that the task is completed\n' + "Start time: " + starttime + " End time: " + endtime + "\n"
print(mailmessage)

