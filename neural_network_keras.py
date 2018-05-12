import numpy, time
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils, to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import backend as K
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score

n_classes = 2
num_rows = 1000

text = pd.read_csv('data/Cleaned_data.csv',nrows=num_rows)
#text = pd.read_csv('data/Training_data_with_neutral.csv',nrows=num_rows)
#text = pd.read_csv('data/Feature_extracted.csv',nrows=num_rows)
#text = pd.read_csv('data/Feature_extracted.csv')
#print text['tfidf']
text = text[text.tweet != ""]
print text


tfidf_data = TfidfVectorizer(use_idf=1,stop_words='english')
tweet_data = tfidf_data.fit_transform(text['tweet'].values.astype('U')).toarray()


x_train, x_test, y_train, y_test =  train_test_split(tweet_data, text['senti'], test_size=0.20, random_state=0)
#print y_train

#print y_train[0],x_train.shape, x_train[0][0], len(x_train[2])

#print y_train[0] == 1
#print y_train[1]

'''
p=0
n=0
for i in range(0,len(y_train)):
	try:
		print i, y_train[i]
		if y_train[i] == 1:
			print "---"
		elif y_train[i] == 0:
			print "---"
		else:
			print "---"
	except:
		print "ddd"
'''


input_length = len(x_train[2])

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

def Evaluation_parameters(y_test, y_pred):
	accuracy = accuracy_score(y_test, y_pred)
	precision = average_precision_score(y_test, y_pred)
	f1score = f1_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	cohen_kappa = cohen_kappa_score(y_test, y_pred)
	Hamming_loss = hamming_loss(y_test, y_pred)
	jaccard_similarity = jaccard_similarity_score(y_test, y_pred)
	Confusion_matrix = confusion_matrix(y_test, y_pred).ravel()

	return accuracy, precision, f1score, recall, cohen_kappa, Hamming_loss, jaccard_similarity, Confusion_matrix

def baseline_model():
	# create model with one hidden layer with rectifier activation function and output layer with softmax activation function
	model = Sequential()
	model.add(Dense(input_length, input_dim=input_length, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(500, input_dim=500, kernel_initializer='normal', activation='relu'))
	model.add(Dense(300, input_dim=500, kernel_initializer='normal', activation='relu'))
	#model.add(Dense(100, input_dim=300, kernel_initializer='normal', activation='relu'))
	model.add(Dense(n_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, fbeta_score])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


start = time.clock()
model = baseline_model()
# Fit the model
model.fit(x_train, Y_train, validation_data=(x_test, Y_test), epochs=5, batch_size=50, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
Time_nn = time.clock() - start

y_pred = model.predict(x_test, batch_size=128)
Y_pred = []

for i in range(0,len(y_pred)):
	if y_pred[i][0] >= y_pred[i][1]:
		a = 1
	else:
		a = -1
	Y_pred.append(a)
"""
for i in range(0,len(y_pred)):
	if y_pred[i][0] == max(y_pred[i]):
		a = 1
	elif y_pred[i][2] ==  max(y_pred[i]):
		a = -1
	else:
		a = 0

	print y_pred[i], a

	Y_pred.append(a)
"""

print Y_pred
accuracy_nn, precision_nn, f1score_nn, recall_nn, cohen_kappa_nn, hamming_loss_nn, jaccard_similarity_nn, Confusion_matrix_nn = Evaluation_parameters(y_test, Y_pred)
ep = [accuracy_nn, precision_nn, f1score_nn, recall_nn, cohen_kappa_nn, hamming_loss_nn, jaccard_similarity_nn, Time_nn, Confusion_matrix_nn]

print ep
