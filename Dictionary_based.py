#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

text = pd.read_csv('data/Data_Integration_Final.csv',nrows=10)
text.drop(text.columns[0],axis=1,inplace=True)

file = open('dictionary_based/positive_words2', 'r') 
a = file.read() 
pos_words = set(a.lower().split('\n'))

file = open('dictionary_based/negative_words.txt', 'r') 
a = file.read() 
neg_words = set(a.split('\n'))

y_pred = []
correct = 0
for i in range(0,len(text['tweet'])):
	a = word_tokenize((str(text['tweet'][i])).lower())
	score = 0
	for each in a:
		each1 = lemmatizer.lemmatize(each)
		each2 = porter.stem(each)
		if ((each1 in pos_words) or (each2 in pos_words)):
			score+=1
		if ((each1 in neg_words) or (each2 in neg_words)):
			score-=1
	if(score > 0):
		y_pred.append(1)
	else:
		y_pred.append(-1)

	print text['tweet'][i],text['senti'][i],score
	if((score>0 and text['senti'][i] == 1) or (score<0 and text['senti'][i] == -1) or (score==0 and text['senti'][i] == 0) ):
		correct += 1

print "Accuracy : ",(float(correct)/len(text))


accuracy_db, precision_db, f1score_db, recall_db, cohen_kappa_db, hamming_loss_db, jaccard_similarity_db, Confusion_matrix_db = Evaluation_parameters(y_test, Y_pred)
accuracy_db = scores[1]
ep = [accuracy_db, precision_db, f1score_db, recall_db, cohen_kappa_db, hamming_loss_db, jaccard_similarity_db, Time_db, Confusion_matrix_db]
print "\n"
Print_Evaluation_parameters(ep)