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
from keras.layers import Convolution1D, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import TensorBoard
import pickle
from keras.models import model_from_json

n_classes = 2
num_rows = 100

#text = pd.read_csv('data/Training_data_with_neutral.csv',nrows=num_rows)
text = pd.read_csv('data/Feature_extracted.csv',nrows=num_rows)
print text.head()

tfidf_data = TfidfVectorizer(use_idf=1,stop_words='english')
tweet_data = tfidf_data.fit_transform(text['tweet']).toarray()


x_train, x_test, y_train, y_test =  train_test_split(tweet_data, text['senti'], test_size=0.20, random_state=0)
rows = x_train.shape[0]
#print rows
input_length = len(x_train[2])

print input_length


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
'''

start = time.clock()

# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(rows, embedding_vecor_length, input_length=input_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))#,input_shape = (rows, input_length)))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(n_classes,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, Y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

scores = model.evaluate(x_test, Y_test, verbose=0)
print scores
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
Time_cnn = time.clock() - start
"""
# serialize model to JSON
model_json = model.to_json()
with open("pickle/CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("pickle/CNN_model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('pickle/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("pickle/CNN_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, Y_test, verbose=0)
print scores
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
"""
print x_test
y_pred = model.predict(x_test, batch_size=64)

Y_pred = []
print y_pred
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

Y_pred = []
for i in range(0,len(y_pred)):
	if y_pred[i][0] >= y_pred[i][1]:
		a = 1
	else:
		a = -1
	Y_pred.append(a)

print Y_pred


def Evaluation_parameters(y_test, y_pred):
	accuracy = accuracy_score(y_test, y_pred)
	precision = average_precision_score(y_test, y_pred)
	f1score = f1_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	cohen_kappa = cohen_kappa_score(y_test, y_pred)
	Hamming_loss = hamming_loss(y_test, y_pred)
	jaccard_similarity = jaccard_similarity_score(y_test, y_pred)
	Confusion_matrix = confusion_matrix(y_test, y_pred).ravel()

accuracy_cnn, precision_cnn, f1score_cnn, recall_cnn, cohen_kappa_cnn, hamming_loss_cnn, jaccard_similarity_cnn, Confusion_matrix_cnn = Evaluation_parameters(y_test, Y_pred)
ep = [accuracy_cnn, precision_cnn, f1score_cnn, recall_cnn, cohen_kappa_cnn, hamming_loss_cnn, jaccard_similarity_cnn, Time_cnn, Confusion_matrix_cnn]
print ep
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,  Input, Activation
from keras.models import Model
from keras.layers.normalization import BatchNormalization

input_shape = (input_length,1)

inp = Input(shape=input_shape)
model = inp
    
model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
model = Flatten()(model)

model = Dense(56)(model)
model = Activation('relu')(model)
model = BatchNormalization()(model)
model = Dropout(0.2)(model)
model = Dense(28)(model)
model = Activation('relu')(model)
model = BatchNormalization()(model)

model = Dense(1)(model)
model = Activation('sigmoid')(model)
    
model = Model(inp, model)
    


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, Y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, Y_test, batch_size=16)


print x_test
y_pred = model.predict(x_test, batch_size=64)

Y_pred = []
print y_pred

for i in range(0,len(y_pred)):
	if y_pred[i][0] >= y_pred[i][1]:
		a = 1
	else:
		a = -1
	Y_pred.append(a)

print Y_pred


def Evaluation_parameters(y_test, y_pred):
	accuracy = accuracy_score(y_test, y_pred)
	precision = average_precision_score(y_test, y_pred)
	f1score = f1_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	cohen_kappa = cohen_kappa_score(y_test, y_pred)
	Hamming_loss = hamming_loss(y_test, y_pred)
	jaccard_similarity = jaccard_similarity_score(y_test, y_pred)
	Confusion_matrix = confusion_matrix(y_test, y_pred).ravel()

accuracy_cnn, precision_cnn, f1score_cnn, recall_cnn, cohen_kappa_cnn, hamming_loss_cnn, jaccard_similarity_cnn, Confusion_matrix_cnn = Evaluation_parameters(y_test, Y_pred)
ep = [accuracy_cnn, precision_cnn, f1score_cnn, recall_cnn, cohen_kappa_cnn, hamming_loss_cnn, jaccard_similarity_cnn, Time_cnn, Confusion_matrix_cnn]
print ep

