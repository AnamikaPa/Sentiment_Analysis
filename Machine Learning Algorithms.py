#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys
import HTMLParser, string, re, itertools, nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np 
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score
import time 
import matplotlib.pyplot as plt
from decimal import Decimal
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils, to_categorical
from keras.layers import Convolution1D, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob


lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

num_rows = 1000
text = pd.read_csv('data/Cleaned_data_without_neutral.csv', nrows= num_rows)
text = text[text.senti != 0] 

from sklearn.preprocessing import label_binarize
#text['senti'] = label_binarize(text['senti'], classes=['1','-1'])

text.drop(text.columns[0],axis=1,inplace=True)

np.seterr(divide='ignore', invalid='ignore')

#print text

"""
lexicon =[]

for l in text['tweet']:
	all_words = word_tokenize(str(l).lower())
	lexicon += list(all_words)
	#print lexicon

	#Stemming
lexicon = [lemmatizer.lemmatize(i.decode('utf-8')) for i in lexicon]
w_counts = Counter(lexicon)

h=40
l=1

#print w_counts

l2 = []
for w in w_counts:
	if w_counts[w] > l:
		l2.append(w)


print "lexicon size:",np.array(l2).shape

a = []

for i in range(0,len(text['tweet'])):
	#print text['tweet'][i]
	word_tokens = word_tokenize(str(text['tweet'][i]).decode('utf-8'))

	a.append(" ".join([w for w in word_tokens if w in l2]))
	#print text['tweet'][i]

text["tfidf"] = a
"""

tfidf_data = TfidfVectorizer(use_idf=1,stop_words='english')
tweet_data = tfidf_data.fit_transform(text['tweet'].values.astype('U')).toarray()

print tweet_data.shape
"""
integration = pd.DataFrame(text).to_csv('data/Feature_extracted.csv')
"""


x_train, x_test, y_train, y_test =  train_test_split(tweet_data, text['senti'], test_size=0.30, random_state=0)

"""
http://scikit-learn.org/stable/modules/model_evaluation.html
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def Evaluation_parameters(y_test, y_pred, name):
	accuracy = accuracy_score(y_test, y_pred)
	precision = average_precision_score(y_test, y_pred)
	f1score = f1_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	cohen_kappa = cohen_kappa_score(y_test, y_pred)
	Hamming_loss = hamming_loss(y_test, y_pred)
	jaccard_similarity = jaccard_similarity_score(y_test, y_pred)
	Confusion_matrix = confusion_matrix(y_test, y_pred)

	plt.figure()
	plot_confusion_matrix(Confusion_matrix, classes=["Positive", "Negative"], title='Confusion matrix of ' + name)
	plt.show()

	return accuracy, precision, f1score, recall, cohen_kappa, Hamming_loss, jaccard_similarity, Confusion_matrix


def Print_Evaluation_parameters(Array):

	print "Accuracy: ",Array[0]
	print "Precision: ",Array[1]
	print "F1 score: ",Array[2]
	print "Recall: ",Array[3]
	print "Cohen kappa score: ",Array[4]
	print "Hamming Loss: ",Array[5]
	print "Jaccard similarity: ",Array[6]
	print "Time: ",Array[7]
	print "Confusion matrix: ",Array[8]
	#print classification_report(y_test, y_pred, target_names=target_names)
	#print classification_report(y_test, y_pred)



# ------------------------------------ Decision Tree ----------------------------------------

start = time.clock()
clf = DecisionTreeClassifier(random_state=0, min_samples_split=2, min_samples_leaf=1)
clf.fit(x_train,y_train)
Time_dt = time.clock() - start

with open('pickle/DecisionTreeClassifier.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/DecisionTreeClassifier.pickle','rb')
clf = pickle.load(pickle_in)

y_pred = clf.predict(x_test)

print "\n\n--------------- Decision Tree ----------------"

accuracy_dt, precision_dt, f1score_dt, recall_dt, cohen_kappa_dt, hamming_loss_dt, jaccard_similarity_dt, Confusion_matrix_dt = Evaluation_parameters(y_test, y_pred,"Decision Tree")
ep = [accuracy_dt, precision_dt, f1score_dt, recall_dt, cohen_kappa_dt, hamming_loss_dt, jaccard_similarity_dt, Time_dt, Confusion_matrix_dt]
Print_Evaluation_parameters(ep)


# ------------------------------------ Random Forest ----------------------------------------

start = time.clock()
clf = RandomForestClassifier(n_estimators=100, random_state=0, min_samples_split=2, min_samples_leaf=1)
clf.fit(x_train,y_train)
Time_rf = time.clock() - start

with open('pickle/RandomForestClassifier.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/RandomForestClassifier.pickle','rb')
clf = pickle.load(pickle_in)

y_pred = clf.predict(x_test)

print "\n\n--------------- Random Forest ----------------"

accuracy_rf, precision_rf, f1score_rf, recall_rf, cohen_kappa_rf, hamming_loss_rf, jaccard_similarity_rf, Confusion_matrix_rf = Evaluation_parameters(y_test, y_pred,"Random Forest")
ep = [accuracy_rf, precision_rf, f1score_rf, recall_rf, cohen_kappa_rf, hamming_loss_rf, jaccard_similarity_rf, Time_rf, Confusion_matrix_rf]
Print_Evaluation_parameters(ep)


# ------------------------------------ Support Vector Classification ----------------------------------------

start = time.clock()
#clf = svm.SVC(decision_function_shape='ovo')
clf = svm.SVC(kernel="linear", random_state = 0)
clf.fit(x_train,y_train)
Time_svc = time.clock() - start

with open('pickle/SVC.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/SVC.pickle','rb')
clf = pickle.load(pickle_in)

y_pred = clf.predict(x_test)

print "\n\n--------------- Support Vector Classification ----------------"

accuracy_svc, precision_svc, f1score_svc, recall_svc, cohen_kappa_svc, hamming_loss_svc, jaccard_similarity_svc, Confusion_matrix_svc = Evaluation_parameters(y_test, y_pred,"Support Vector Classification")
ep = [accuracy_svc, precision_svc, f1score_svc, recall_svc, cohen_kappa_svc, hamming_loss_svc, jaccard_similarity_svc, Time_svc, Confusion_matrix_svc]
Print_Evaluation_parameters(ep)


# ------------------------------------ Gaussian Naive Bayes ----------------------------------------

start = time.clock()
clf = GaussianNB()
clf.fit(x_train,y_train)
Time_gnb = time.clock() - start

with open('pickle/GaussianNB.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/GaussianNB.pickle','rb')
clf = pickle.load(pickle_in)

y_pred = clf.predict(x_test)

print "\n\n--------------- Gaussian Naive Bayes ----------------"

accuracy_gnb, precision_gnb, f1score_gnb, recall_gnb, cohen_kappa_gnb, hamming_loss_gnb, jaccard_similarity_gnb, Confusion_matrix_gnb = Evaluation_parameters(y_test, y_pred,"Gaussian Naive Bayes")
ep = [accuracy_gnb, precision_gnb, f1score_gnb, recall_gnb, cohen_kappa_gnb, hamming_loss_gnb, jaccard_similarity_gnb, Time_gnb, Confusion_matrix_gnb]
Print_Evaluation_parameters(ep)


# ------------------------------------ Multinomial Naive Bayes ----------------------------------------

start = time.clock()
clf = MultinomialNB()
clf.fit(x_train,y_train)
Time_mnb = time.clock() - start

with open('pickle/MultinomialNB.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/MultinomialNB.pickle','rb')
clf = pickle.load(pickle_in)
y_pred = clf.predict(x_test)

print "\n\n--------------- Multinomial Naive Bayes ----------------"

accuracy_mnb, precision_mnb, f1score_mnb, recall_mnb, cohen_kappa_mnb, hamming_loss_mnb, jaccard_similarity_mnb, Confusion_matrix_mnb = Evaluation_parameters(y_test, y_pred, "Multinomial Naive Bayes")
ep = [accuracy_mnb, precision_mnb, f1score_mnb, recall_mnb, cohen_kappa_mnb, hamming_loss_mnb, jaccard_similarity_mnb, Time_mnb, Confusion_matrix_mnb]
Print_Evaluation_parameters(ep)


# ------------------------------------ Logistic Regression ----------------------------------------

start = time.clock()
clf = linear_model.LogisticRegression()
clf.fit(x_train,y_train)
Time_lr = time.clock() - start

with open('pickle/LogisticRegression.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/LogisticRegression.pickle','rb')
clf = pickle.load(pickle_in)
y_pred = clf.predict(x_test)

print "\n\n--------------- Logistic Regression ----------------"

accuracy_lr, precision_lr, f1score_lr, recall_lr, cohen_kappa_lr, hamming_loss_lr, jaccard_similarity_lr, Confusion_matrix_lr = Evaluation_parameters(y_test, y_pred, "Logistic Regression")
ep = [accuracy_lr, precision_lr, f1score_lr, recall_lr, cohen_kappa_lr, hamming_loss_lr, jaccard_similarity_lr, Time_lr, Confusion_matrix_lr]
Print_Evaluation_parameters(ep)


# ------------------------------------ Multi layer perceptron Classifier ----------------------------------------

start = time.clock()
clf = MLPClassifier(hidden_layer_sizes=(100,100,100,), random_state=1)
clf.fit(x_train,y_train)
Time_mlp = time.clock() - start

with open('pickle/MLPClassifier.pickle','wb') as f:
	pickle.dump(clf,f)

pickle_in = open('pickle/MLPClassifier.pickle','rb')
clf = pickle.load(pickle_in)
y_pred = clf.predict(x_test)

print "\n\n--------------- Multi layer perceptron Classifier ----------------"

accuracy_mlp, precision_mlp, f1score_mlp, recall_mlp, cohen_kappa_mlp, hamming_loss_mlp, jaccard_similarity_mlp, Confusion_matrix_mlp = Evaluation_parameters(y_test, y_pred, "Multi layer perceptron Classifier ")
ep = [accuracy_mlp, precision_mlp, f1score_mlp, recall_mlp, cohen_kappa_mlp, hamming_loss_mlp, jaccard_similarity_mlp, Time_mlp, Confusion_matrix_mlp]
Print_Evaluation_parameters(ep)



# ------------------------------------ Neural Network ----------------------------------------
input_length = len(x_train[2])
n_classes = 2

Y_train = []
Y_test = []


for i in range(0,len(y_train)):
	try:
		if(n_classes == 3):
			if y_train[i] == 0:
				a = [0,1,0]
			elif y_train[i] == 1:
				a= [1,0,0]
			else:
				a = [0,0,1]
		else:
			if y_train[i] == 1:
				a = [1,0]
			else:
				a = [0,1]
	except:
		if(n_classes == 3):
			a = [1,0,0]
		else:
			a = [1,0]

	Y_train.append(a)


for i in range(0,len(y_test)):
	try:
		if(n_classes == 3):
			if y_test[i] == 0:
				a = [0,1,0]
			elif y_test[i] == 1:
				a= [1,0,0]
			else:
				a = [0,0,1]
		else:
			if y_test[i] == 1:
				a = [1,0]
			else:
				a = [0,1]
	except:
		if(n_classes == 3):
			a = [1,0,0]
		else:
			a = [1,0]

	Y_test.append(a)


print "\n\n--------------- Neural Network ----------------"


def baseline_model():
	
	# create model with one hidden layer with rectifier activation function and output layer with softmax activation function
	model = Sequential()
	model.add(Dense(input_length, input_dim=input_length, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(500, input_dim=500, kernel_initializer='normal', activation='relu'))
	model.add(Dense(500, input_dim=input_length, kernel_initializer='normal', activation='relu'))
	model.add(Dense(300, input_dim=500, kernel_initializer='normal', activation='relu'))
	model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fbeta_score])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	return model


start = time.clock()
model = baseline_model()
# Fit the model
model.fit(np.array(x_train), np.array(Y_train), validation_data=(np.array(x_test), np.array(Y_test)), epochs=5, batch_size=50, verbose=2)
# Final evaluation of the model
scores = model.evaluate(np.array(x_test), np.array(Y_test), verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
Time_nn = time.clock() - start

y_pred = model.predict(x_test, batch_size=128)
Y_pred = []
for i in range(0,len(y_pred)):
	if y_pred[i][0] >= y_pred[i][1]:
		a = 1
	else:
		a = -1
	Y_pred.append(a)

print "\n\n--------------- Neural Network ----------------"
accuracy_nn, precision_nn, f1score_nn, recall_nn, cohen_kappa_nn, hamming_loss_nn, jaccard_similarity_nn, Confusion_matrix_nn = Evaluation_parameters(y_test, Y_pred, "Neural Network")
#accuracy_nn = scores[1]
ep = [accuracy_nn, precision_nn, f1score_nn, recall_nn, cohen_kappa_nn, hamming_loss_nn, jaccard_similarity_nn, Time_nn, Confusion_matrix_nn]
print "\n"
Print_Evaluation_parameters(ep)


"""
# ------------------------------------ Convolution Neural Network ----------------------------------------

print "\n\n--------------- Convolutional Neural Network ----------------"

start = time.clock()

# Using embedding from Keras
embedding_vecor_length = 30
cnn_model = Sequential()
cnn_model.add(Embedding(num_rows, embedding_vecor_length, input_length=input_length))
#cnn_model.add(Embedding(input_length, input_length, input_length=input_length))

# Convolutional model (3x conv, flatten, 2x dense)
cnn_model.add(Convolution1D(64, 3, padding='same'))
cnn_model.add(Convolution1D(32, 3, padding='same'))
cnn_model.add(Convolution1D(16, 3, padding='same'))
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(180,activation='sigmoid'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(2,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

cnn_model.fit(np.array(x_train), np.array(Y_train), epochs=5, callbacks=[tensorBoardCallback], batch_size=64)
scores = cnn_model.evaluate(np.array(x_test), np.array(Y_test), verbose=0)
print scores
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
Time_cnn = time.clock() - start

y_pred = cnn_model.predict(x_test, batch_size=128)
Y_pred = []
for i in range(0,len(y_pred)):
	if y_pred[i][0] >= y_pred[i][1]:
		a = 1
	else:
		a = -1
	Y_pred.append(a)


accuracy_cnn, precision_cnn, f1score_cnn, recall_cnn, cohen_kappa_cnn, hamming_loss_cnn, jaccard_similarity_cnn, Confusion_matrix_cnn = Evaluation_parameters(y_test, Y_pred, "Convolution Neural Network")
#accuracy_cnn = scores[1]
ep = [accuracy_cnn, precision_cnn, f1score_cnn, recall_cnn, cohen_kappa_cnn, hamming_loss_cnn, jaccard_similarity_cnn, Time_cnn, Confusion_matrix_cnn]
print "\n"
Print_Evaluation_parameters(ep)
"""

# ------------------------------------ Dictionary Based ----------------------------------------

start = time.clock()

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
	

Time_db = time.clock() - start


#print "Accuracy : ",(float(correct)/len(text))

print "\n\n--------------- Dictionary Based ----------------"

accuracy_db, precision_db, f1score_db, recall_db, cohen_kappa_db, hamming_loss_db, jaccard_similarity_db, Confusion_matrix_db = Evaluation_parameters(text['senti'], y_pred, "Dictionary Based")
ep = [accuracy_db, precision_db, f1score_db, recall_db, cohen_kappa_db, hamming_loss_db, jaccard_similarity_db, Time_db, Confusion_matrix_db]
Print_Evaluation_parameters(ep)


# ------------------------------------ Dictionary Based with Score ----------------------------------------

start = time.clock()
sentiment = pd.read_csv('dictionary_based/sentiment_score.txt',sep='\t')
sentiment.drop(sentiment.columns[[1,5]],axis=1,inplace=True)
#print sentiment.head()


senti = dict()

for i in range(0,len(sentiment)):
	if(sentiment['PosScore'][i]!=0 or sentiment['NegScore'][i]!=0):
		a = sentiment['SynsetTerms'][i].split(" ")
		score = sentiment['PosScore'][i] - sentiment['NegScore'][i]
		for w in a:
			#print w, score
			senti[w] = score

#print senti

y_pred = []
correct = 0
for i in range(0,len(text['tweet'])):
	a = word_tokenize((str(text['tweet'][i])).lower())
	score = 0
	for each in a:
		each1 = lemmatizer.lemmatize(each)
		each2 = porter.stem(each)
		if (each1 in senti):
			score+=senti[each1]
		if (each2 in senti):
			score+=senti[each2]
	
	if(score > 0):
		y_pred.append(1)
	else:
		y_pred.append(-1)
	
Time_dbs = time.clock() - start

#print "Accuracy : ",(float(correct)/len(text))

print "\n\n--------------- Dictionary Based with Score ----------------"

accuracy_dbs, precision_dbs, f1score_dbs, recall_dbs, cohen_kappa_dbs, hamming_loss_dbs, jaccard_similarity_dbs, Confusion_matrix_dbs = Evaluation_parameters(text['senti'], y_pred, "Dictionary Based with Score")
ep = [accuracy_dbs, precision_dbs, f1score_dbs, recall_dbs, cohen_kappa_dbs, hamming_loss_dbs, jaccard_similarity_dbs, Time_dbs, Confusion_matrix_dbs]
Print_Evaluation_parameters(ep)


# ------------------------------------ Text Blob Sentiment API ----------------------------------------


start = time.clock()

y_pred = []
correct = 0
for i in range(0,len(text['tweet'])):
	analysis = TextBlob((str(text['tweet'][i])).lower())
	
	if(analysis.sentiment.polarity >= 0):
		y_pred.append(1)
	else:
		y_pred.append(-1)
Time_tb_api = time.clock() - start

print "\n\n--------------- Support Vector Classification ----------------"
accuracy_tb_api, precision_tb_api, f1score_tb_api, recall_tb_api, cohen_kappa_tb_api, hamming_loss_tb_api, jaccard_similarity_tb_api, Confusion_matrix_tb_api = Evaluation_parameters(text['senti'], y_pred, "Text Blob Sentiment API")
ep = [accuracy_tb_api, precision_tb_api, f1score_tb_api, recall_tb_api, cohen_kappa_tb_api, hamming_loss_tb_api, jaccard_similarity_tb_api, Time_tb_api, Confusion_matrix_tb_api]
Print_Evaluation_parameters(ep)



# -------------------------------- GRAPHS -------------------------------------------

def normalization(value , min, max):
	max = Decimal(max)
	min = Decimal(min)
	value = Decimal(value)
	return (value - min)/(max - min)


def bar_chart(y_pos, performance, xlabel, ylabel, title, objects):
	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()


#------------------------- Accuracy without Normalization ----------------------------- 

#objects = ('Dcsn Tree', 'Rand Frst', 'SVC', 'GaussNB', 'MultiNB', 'Logi Rgrsn', 'MLP', 'NN', 'CNN', 'DB', 'DBS')
objects = ('Dcsn Tree', 'Rand Frst', 'SVC', 'GaussNB', 'MultiNB', 'Logi Rgrsn', 'MLP', 'NN', 'DB', 'DBS')
y_pos = np.arange(len(objects))
#performance1 = [accuracy_dt, accuracy_rf, accuracy_svc, accuracy_gnb, accuracy_mnb, accuracy_lr, accuracy_mlp, accuracy_nn, accuracy_cnn, accuracy_db, accuracy_dbs]
performance1 = [accuracy_dt, accuracy_rf, accuracy_svc, accuracy_gnb, accuracy_mnb, accuracy_lr, accuracy_mlp, accuracy_nn, accuracy_db, accuracy_dbs]

 
bar_chart(y_pos, performance1, "Machine Learning Algorithms", "Accuracy(0-1)", "Comparision of Accuracy without normalization", objects)



#------------------------- Accuracy with Normalization ----------------------------- 

performance = [""]*10
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Accuracy", "Comparision of Accuracy with normalization", objects)


#------------------------- Precision with Normalization ----------------------------- 

#performance1 = [precision_dt, precision_rf, precision_svc, precision_gnb, precision_mnb, precision_lr, precision_mlp, precision_nn, precision_cnn, precision_db,precision_dbs]
performance1 = [precision_dt, precision_rf, precision_svc, precision_gnb, precision_mnb, precision_lr, precision_mlp, precision_nn, precision_db,precision_dbs]

max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Precision", "Comparision of Precision with normalization", objects)



#------------------------- F1 score with Normalization ----------------------------- 

#performance1 = [f1score_dt, f1score_rf, f1score_svc, f1score_gnb, f1score_mnb, f1score_lr, f1score_mlp, f1score_nn, f1score_cnn, f1score_db, f1score_dbs]
performance1 = [f1score_dt, f1score_rf, f1score_svc, f1score_gnb, f1score_mnb, f1score_lr, f1score_mlp, f1score_nn, f1score_db, f1score_dbs]
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "F1 score", "Comparision of F1 score with normalization", objects)


#------------------------- Recall with Normalization ----------------------------- 

#performance1 = [recall_dt, recall_rf, recall_svc, recall_gnb, recall_mnb, recall_lr, recall_mlp, recall_nn, recall_cnn, recall_db, recall_dbs]
performance1 = [recall_dt, recall_rf, recall_svc, recall_gnb, recall_mnb, recall_lr, recall_mlp, recall_nn, recall_db, recall_dbs]
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Recall", "Comparision of Recall with normalization", objects)


#------------------------- Time with Normalization ----------------------------- 

#performance1 = [Time_dt, Time_rf, Time_svc, Time_gnb, Time_mnb, Time_lr, Time_mlp, Time_nn, Time_cnn, Time_db, Time_dbs]
performance1 = [Time_dt, Time_rf, Time_svc, Time_gnb, Time_mnb, Time_lr, Time_mlp, Time_nn, Time_db, Time_dbs]
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Time", "Comparision of Time with normalization", objects)


#------------------------- Jaccard similarity score with Normalization ----------------------------- 

#performance1 = [jaccard_similarity_dt, jaccard_similarity_rf, jaccard_similarity_svc, jaccard_similarity_gnb, jaccard_similarity_mnb, jaccard_similarity_lr, jaccard_similarity_mlp, jaccard_similarity_nn, jaccard_similarity_cnn, jaccard_similarity_db, jaccard_similarity_dbs]
performance1 = [jaccard_similarity_dt, jaccard_similarity_rf, jaccard_similarity_svc, jaccard_similarity_gnb, jaccard_similarity_mnb, jaccard_similarity_lr, jaccard_similarity_mlp, jaccard_similarity_nn, jaccard_similarity_db, jaccard_similarity_dbs]
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Jaccard similarity score", "Comparision of Jaccard similarity score with normalization", objects)


#------------------------- Hamming Loss with Normalization ----------------------------- 

#performance1 = [hamming_loss_dt, hamming_loss_rf, hamming_loss_svc, hamming_loss_gnb, hamming_loss_mnb, hamming_loss_lr, hamming_loss_mlp, hamming_loss_nn, hamming_loss_cnn, hamming_loss_db, hamming_loss_dbs]
performance1 = [hamming_loss_dt, hamming_loss_rf, hamming_loss_svc, hamming_loss_gnb, hamming_loss_mnb, hamming_loss_lr, hamming_loss_mlp, hamming_loss_nn, hamming_loss_db, hamming_loss_dbs]
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Hamming Loss", "Comparision of Hamming Loss with normalization", objects)


#------------------------- Cohen kappa score with Normalization ----------------------------- 

#performance1 = [cohen_kappa_dt, cohen_kappa_rf, cohen_kappa_svc, cohen_kappa_gnb, cohen_kappa_mnb, cohen_kappa_lr, cohen_kappa_mlp, cohen_kappa_nn, cohen_kappa_cnn, cohen_kappa_db, cohen_kappa_dbs]
performance1 = [cohen_kappa_dt, cohen_kappa_rf, cohen_kappa_svc, cohen_kappa_gnb, cohen_kappa_mnb, cohen_kappa_lr, cohen_kappa_mlp, cohen_kappa_nn,  cohen_kappa_db, cohen_kappa_dbs]
max1 = max(performance1)
min1 = min(performance1)

for i in range(0,len(performance1)):
	performance[i] = normalization(performance1[i], min1, max1)

bar_chart(y_pos, performance, "Machine Learning Algorithms", "Cohen kappa score", "Comparision of Cohen kappa score with normalization", objects)


print "\n\n"