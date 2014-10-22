import re, io, json

with open("data/dictionary/source/subjclueslen1-HLTEMNLP05.tff") as dicobject:
    
    dicdata = dicobject.readlines()
    dicItems = []
    for line in dicdata:
        dicline = re.split('\s+|\=+', line)
        dicItems.append({'word': dicline[5], 'type': dicline[1], 'priorpolarity': dicline[11]})
        
    with io.open('data/dictionary/mydictionary_6bins.json', 'w', encoding='utf-8') as outfile:
        outfile.write(unicode(json.dumps(dicItems, ensure_ascii=False)))

with open("data/dictionary/source/neg.txt") as negdicobject, open("data/dictionary/source/pos.txt") as posdicobject, open("data/dictionary/source/negation.txt") as negationdicobject:
    
    negdicdata = negdicobject.readlines()
    posdicdata = posdicobject.readlines()
    negationdicdata = negationdicobject.readlines()
    dicItems = []
    for line in negdicdata:
        word = line.split()[0]
        normed = re.sub('[^a-z]', '', word.lower())
        if normed:
            dicItems.append({'word': normed, 'type': 'none', 'priorpolarity': 'negative'})
    for line in posdicdata:
        word = line.split()[0]
        normed = re.sub('[^a-z]', '', word.lower())
        if normed:
            dicItems.append({'word': normed, 'type': 'none', 'priorpolarity': 'positive'})
    for line in negationdicdata:
        word = line.split()[0]
        normed = re.sub('[^a-z]', '', word.lower())
        if normed:
            if normed not in dicItems:
                dicItems.append({'word': normed, 'type': 'none', 'priorpolarity': 'negation'})
        
    with io.open('data/dictionary/mydictionary_3bins.json', 'w', encoding='utf-8') as outfile:
        outfile.write(unicode(json.dumps(dicItems, ensure_ascii=False)))
