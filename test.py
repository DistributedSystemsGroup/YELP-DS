import re, io, json

with open("data/dictionary/source/SentiWordNet_3.0.0.txt") as dicobject:
    dicdata = dicobject.readlines()
    length = str(len(dicdata))
dicItems = []
i = 0
print("Step 1\n")
try:    
    for line in dicdata:
        i +=1
        print ("line " + str(i) + " of " + length + '\n')
        dicline = re.split('\s+|\=+', line)
        for word in dicline:            
            if "#" in word:
                test = re.split('\#+', word)
                if len(test)==2 and test[0] != '':
                    word = test[0]
                    quan = float(test[1])
                    pos = quan * float(dicline[2])
                    neg = quan * float(dicline[3])
                    #print(word + " " + str(quan) + " " + str(pos) + " " + str(neg))
                    if (len(dicItems) == 0):
                        dicItems.append({'word': word, 'quantity': quan, 'pos': pos, 'neg': neg})
                    isFound = False
                    for idx, item in enumerate(dicItems):                        
                        if item["word"] == word:
                            newQuan = quan + int(item["quantity"])
                            newPos = pos + float(item["pos"])
                            newNeg = neg + float(item["neg"])
                            dicItem = {'word': word, 'quantity': newQuan, 'pos': newPos, 'neg': newNeg}
                            dicItems[idx] = dicItem
                            isFound = True
                    if isFound == False:
                        dicItems.append({'word': word, 'quantity': quan, 'pos': pos, 'neg': neg})
                            
                            
except Exception:
    print line
    print len(test)
    print(test[0])
    print(test[0])
    raise
print("Step 2\n")
for idx, item in enumerate(dicItems):
    newPosScore = float(item["pos"]) / float(item["quantity"])
    newNegScore = float(item["neg"]) / float(item["quantity"])
    newDicItem = {'word': item["word"], 'pos': newPosScore, 'neg': newNegScore}
    dicItems[idx] = newDicItem

print(dicItems)
print("Step 3\n")
with io.open('data/dictionary/mydictionary_2bins.json', 'w', encoding='utf-8') as outfile:
    outfile.write(unicode(json.dumps(dicItems, ensure_ascii=False)))

with open("data/dictionary/mydictionary_2bins.json") as dicObject2Bins:
    dicData2Bins = json.load(dicObject2Bins)
dicItems = []
i = 0
for item in dicData2Bins:
    if float(item["pos"]) == 0.0 and float(item["neg"]) == 0.0:
        continue
        print("skip")
    if "-" in item["word"] or "_" in item["word"]:
        continue
        print("-------------------------------------------------------------------------------------")
    dicItems.append(item)
    i += 1
    print(str(i))

with io.open('data/dictionary/mydictionary_2bins_reduced.json', 'w', encoding='utf-8') as outfile:
    outfile.write(unicode(json.dumps(dicItems, ensure_ascii=False)))
