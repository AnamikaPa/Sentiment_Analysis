#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text = pd.read_csv('data/Data_Integration_Test.csv',nrows=10)
text.drop(text.columns[0],axis=1,inplace=True)
print text


sentiment = pd.read_csv('dictionary_based/sentiment_score.txt',sep='\t')
sentiment.drop(sentiment.columns[[1,5]],axis=1,inplace=True)
print sentiment.head()


emoji = pd.read_csv('dictionary_based/emoji_score',sep='\t')
emoji.drop(emoji.columns[[3,4,5,6,7,9]],axis=1,inplace=True)
print emoji.head()



emo = dict()
for i in range(0,len(emoji)):
	emo[emoji['char'][i]] = emoji['score'][i]

print emo

senti = dict()

for i in range(0,len(sentiment)):
	if(sentiment['PosScore'][i]!=0 or sentiment['NegScore'][i]!=0):
		a = sentiment['SynsetTerms'][i].split(" ")
		score = sentiment['PosScore'][i] - sentiment['NegScore'][i]
		for w in a:
			#print w, score
			senti[w] = score

#print senti

correct = 0
for i in range(0,len(text['tweet'])):
	a = word_tokenize((text['tweet'][i]).lower())
	
	score = 0
	for each in a:
		if(each in emo):
			score += emo[each]
			print "------------"
		each1 = lemmatizer.lemmatize(each)
		each2 = porter.stem(each)
		if (each1 in senti):
			score+=senti[each1]
		if (each2 in senti):
			score+=senti[each2]
	
	#print text['tweet'][i],text['senti'][i],score
	if((score>0 and text['senti'][i] == 1) or (score<0 and text['senti'][i] == -1) or (score==0 and text['senti'][i] == 0) ):
		correct +=1

print "Accuracy : ",(float(correct)/len(text))

