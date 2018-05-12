import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score
import pickle
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn import linear_model

print "Loding Data......."
text = pd.read_csv('data/Cleaned_data_with_neutral.csv')

print "\nEnter the name of file you wanted to read from: "
file = raw_input()


test = pd.read_csv(file)
num_rows = len(text)
combined = pd.concat([text['tweet'],test['tweet']])
length = len(combined)

a = combined[num_rows:]
b = combined[0:num_rows]

tfidf_data = TfidfVectorizer(use_idf=1,stop_words='english')
tweet_data1 = tfidf_data.fit_transform(combined.values.astype('U')).toarray()


length = len(tweet_data1)

test_data = tweet_data1[num_rows:]
tweet_data = tweet_data1[0:num_rows]

x_train, x_test, y_train, y_test =  train_test_split(tweet_data, text['senti'], test_size=0.20, random_state=0)


print "\nTraining Model .............. "
clf = linear_model.LogisticRegression()
clf.fit(x_train,y_train)

with open('pickle/LogisticRegression.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/LogisticRegression.pickle','rb')
clf = pickle.load(pickle_in)

y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print "Accuracy: ",accuracy

y_real_pred = clf.predict(test_data)
pos = neg = neu = 0

for i in y_real_pred:
	if i == 1:
		pos+=1
	elif i == 0:
		neu+=1
	else:
		neg+=1		

print "\nPositive Tweets: ",pos
print "Negative Tweets: ",neg
print "Neutral Tweets: ",neu

pos = Decimal(pos)
neg = Decimal(neg)
neu = Decimal(neu)
Rating = 5*((pos+neu)/(pos+neu+neg))

Rating = str(round(Rating, 2))
print "\nRating out of 5: ",Rating

labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos,neg,neu]
colors = ['yellowgreen', 'lightcoral', 'gold']
explode = (0.001, 0.001, 0.001)  # explode 1st slice
 
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.tight_layout()
plt.title('\n Ratings: '+ str(Rating)+' / 5')
plt.axis('equal')
plt.show()
