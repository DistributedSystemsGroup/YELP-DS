#! /usr/bin/env python

import io, os
import random

numberOfData = 1125458
numberOfSample = 200

inputfile = 'data/input/source/yelp_academic_dataset_review.json'
outputfile = 'data/input/200samples.json'

randomSelectionList = random.sample(xrange(0, numberOfData), numberOfSample)

if os.path.isfile(outputfile):
    os.remove(outputfile)
with io.open(outputfile, 'a', encoding='utf-8') as outfile:
    with open(inputfile) as fileobject:
       for i, line in enumerate(fileobject):
           if i in randomSelectionList:
               outfile.write(unicode(line))
