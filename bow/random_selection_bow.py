#! /usr/bin/env python

import io, os
import random, json

numberOfData = 1125458
numberOfSample = 10000

inputfile = '../data/input/source/yelp_academic_dataset_review.json'
outputfile5star = 'data/input/5StarsSamples.json'
outputfile4star = 'data/input/4StarsSamples.json'
outputfile3star = 'data/input/3StarsSamples.json'
outputfile2star = 'data/input/2StarsSamples.json'
outputfile1star = 'data/input/1StarsSamples.json'

randomSelectionList = random.sample(xrange(0, numberOfData), numberOfSample)

for i in xrange(1,6):
    if os.path.isfile('data/input' + str(i) + 'StarsSamples.json'):
        os.remove('data/input' + str(i) + 'StarsSamples.json')

with open(outputfile5star, 'a') as _5starfile, open(outputfile4star, 'a') as _4starfile, open(outputfile3star, 'a') as _3starfile, open(outputfile2star, 'a') as _2starfile, open(outputfile1star, 'a') as _1starfile:
    with open(inputfile) as inputfileobject:
        for i, line in enumerate(inputfileobject):
           if i in randomSelectionList:
               if line == '\n':
                    break
               data = json.loads(line)
               if data["text"] =="":
                   break
               if (data["stars"] == 5):
                   _5starfile.write((data["text"].encode('utf-8')))
               elif (data["stars"] == 4):
                   _4starfile.write((data["text"].encode('utf-8')))
               elif (data["stars"] == 3):
                   _3starfile.write((data["text"].encode('utf-8')))
               elif (data["stars"] == 2):
                   _2starfile.write((data["text"].encode('utf-8')))
               elif (data["stars"] == 1):
                   _1starfile.write((data["text"].encode('utf-8')))

