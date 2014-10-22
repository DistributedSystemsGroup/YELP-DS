import re
import io, json

import smtplib
from email.mime.text import MIMEText

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

def sendMail(mailmessage):
    # Define email addresses to use
    addr_to   = 'caikehe@gmail.com'
    addr_from = 'walleve@mail.com'
 
    # Define SMTP email server details
    smtp_server = 'smtp.mail.com'
    smtp_user   = 'walleve@mail.com'
    smtp_pass   = 'test123456'

    # Construct email
    msg = MIMEText(mailmessage)
    msg['To'] = addr_to
    msg['From'] = addr_from
    msg['Subject'] = 'Notification Email'

    # Send the message via an SMTP server
    s = smtplib.SMTP(smtp_server)
    s.login(smtp_user,smtp_pass)
    s.sendmail(addr_from, addr_to, msg.as_string())
    s.quit()        

class Histogram():
    def run(self, filename, dictionary):
        
        starttime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        
        with open("data/dictionary/mydictionary_3bins.json") as dicObject3Bins, open("data/dictionary/mydictionary_6bins.json") as dicObject6Bins, open(filename) as fileObject:    
            dicData3Bins = json.load(dicObject3Bins)
            dicData6Bins = json.load(dicObject6Bins)
            listOfHistogramAndRating = []
            
            for i, line in enumerate(fileObject):
                if line == '\n':
                    break
                #if i not in xrange(0, 100000):
                    #break
                print(i)

                data = json.loads(line)
                #Descripton [0-spos, 1-wpos, 2-sneu, 3-wneu, 4-sneg, 5-wneg, 6-pos, 7-neg, 8-negation, 9-len, 10-day, 11-vote]
                result =    [0     , 0     , 0     , 0     , 0     , 0     , 0    , 0    , 0         , 0    , 0     , 0      ]
                length =  0
                for word in words(data["text"]):
                    length += 1
                    for item in dicData3Bins:
                        if word == item["word"]:
                            buildingHistogram3Bins(item["priorpolarity"], result)
                    for item in dicData6Bins:
                        if word == item["word"]:
                            buildingHistogram6Bins(item["priorpolarity"], result)

                buildingHistogramDay(data["date"], result)

                buildingHistogramVote(data["votes"], result)
                
                buildingHistogramLength(length, result)
                
                #Fearures lesection
                final_result = []
                final_result.append(result[6])
                final_result.append(result[7])
                listOfHistogramAndRating.append({'rating': data["stars"], 'histogram': final_result})
                

            with io.open('data/output/histogram.json', 'w', encoding='utf-8') as outfile:
                outfile.write(unicode(json.dumps(listOfHistogramAndRating, ensure_ascii=False)))

        print "FINISHED"

        endtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())       

        mailmessage = 'This is a notification email to show that the task is completed\n' + "Start time: " + starttime + " End time: " + endtime + "\n"
        print(mailmessage)
        #sendMail(mailmessage)    


if __name__ == "__main__":
    Histogram().run("data/input/1000samples.json")
