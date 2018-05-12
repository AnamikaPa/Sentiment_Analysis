import re, string, csv, json
import tweepy, enchant, nltk
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
from nltk.corpus import stopwords
from tweepy import OAuthHandler, Stream
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem.porter import PorterStemmer
from replacers import *
from sklearn import linear_model
import pickle
from tweepy.streaming import StreamListener
import json, time, sys, string

global str

lemmatizer = WordNetLemmatizer()
slang_replacer = CsvWordReplacer('dictionary_based/slang1.csv')
words = set(nltk.corpus.words.words())
printable = set(string.printable)
porter = PorterStemmer()


#consumer key, consumer secret, access token, access secret.
ckey=""
csecret=""
atoken=""
asecret=""

track_list = []

while True:
    print "\nEnter the topics you want to search: "

    a = raw_input()
    track_list.append(a)

    print "\nWant to enter more topic? (Y/n) "

    a = raw_input()
    if a == "N" or a == "n" :
        break


print "\nEnter the name of file you want to save data with location: "
file = raw_input()

text_doc = open(file, 'w')
text_doc.write(',tweet\n') 
text_doc.close()

print "\n\nProcessing Stated....."

class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        str = json.dumps(all_data,indent=4)
        tweet = all_data['text']
        lang = all_data['lang']
        print(tweet)
        
        try:
            if lang == 'en':
                tweet = filter(lambda x: x in printable, tweet)

                a = re.sub(r"http\S+", "", tweet)   # Removing URLs
                a = RegexpReplacer().replace(a) 
                a = a.lower()             
                

                print(a)        

                word_tokens = word_tokenize(a)                      # Tokenization
                stop_words = set(stopwords.words('english'))

                
                s = []
                for a in word_tokens:
                    a = slang_replacer.replace(a)
                    s.append(a)
                    #print s
                
                s = " ".join(s)
                #s = s.translate(str.maketrans('','',string.punctuation))
                s = s.translate(string.punctuation)


                print "-----------------------------" 
                print(s)

                word_tokens = word_tokenize(s) 
                s = []
                for a in word_tokens:
                    a = RepeatReplacer().replace(a)         
                    a = a.split(" ")
                    for i in a:
                        i = SpellingReplacer().replace(i)
                        i = i.split(" ")
                        for j in i:
                            j = lemmatizer.lemmatize(j)
                            s.append(j)

                print s
                s = AntonymReplacer().replace_negations(s)
                
                p = []
                for a in s:
                    if a not in stop_words:
                        if a.isalpha():
                            p.append(a)  


                #p = " ".join(p)
                p = " ".join(w for w in p if w in words)
                p = re.sub(r'\b\w{1,2}\b', '', p)            #remove word whose length is less than 3
                a = word_tokenize(p)
                length = len(a)

                p = " ".join(a)

                print "---------------------------------------------------------------"
                print p

                if length>=3:
                    d = {'tweet': [p]}
                    df = pd.DataFrame(data=d)
                    #print df
              
                    with open(file, 'a') as f:
                        df.to_csv(f, header=False)

        except:
            print "Error"
            
        return True

    def on_error(self, status):
        print (status)  

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=track_list)
