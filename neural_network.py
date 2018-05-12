
#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 3

batch_size = 32

num_rows = 1000
#text = pd.read_csv('data/Feature_extracted.csv',nrows=num_rows)
text = pd.read_csv('data/Feature_extracted.csv',nrows = num_rows)
#print text['tfidf']
print text


tfidf_data = TfidfVectorizer(use_idf=1,stop_words='english')
tweet_data = tfidf_data.fit_transform(text['tweet']).toarray()


x_train, x_test, y_train, y_test =  train_test_split(tweet_data, text['senti'], test_size=0.20, random_state=0)
#print y_train

#print y_train,x_train.shape, x_train[0][0], len(x_train[2])


input_length = len(x_train[2])


total_batches = int(num_rows/batch_size)
hm_epochs = 1

x = tf.placeholder('float')
y = tf.placeholder('float')


hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([input_length, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,"model.ckpt")
            epoch_loss = 1

            batch_x = []
            batch_y = []
            batches_run = 0
            for i in range(0,len(x_train)-batch_size):
                batch_x.append(list(x_train[i]))
                if(n_classes == 3):
                    if y_train[i] == 0:
                        a = [0,1,0]
                    elif y_train[i] == 1:
                        a = [1,0,0]
                    else:
                        a = [0,0,1]
                else:
                    if y_train[i] == 1:
                        a = [1,0]
                    else:
                        a = [0,1]

                batch_y.append(a)
                if len(batch_x) >= batch_size:
                    _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x), y: np.array(batch_y)})
                    epoch_loss += c
                    batch_x = []
                    batch_y = []
                    batches_run +=1
                    print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)

            #saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1

train_neural_network(x)


def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        """
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
        """ 
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        """
        for line in f:
            try:
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
        """
        #print('Tested',counter,'samples.')
        for i in range(0,len(y_test)):
            try:
                if(n_classes == 3):
                    if y_test[i] == 0:
                        a = [0,1,0]
                    elif y_test[i] == 1:
                        a = [1,0,0]
                    else:
                        a = [0,0,1]
                else:
                    if y_test[i] == 1:
                        a = [1,0]
                    else:
                        a = [0,1]

                labels.append(a)
                
            except:
                labels.append([0,1,0])


        test_x = np.array(x_test)
        test_y = np.array(labels)
        print '\nAccuracy with Neural Network:',accuracy.eval({x:test_x, y:test_y}),"\n\n"

test_neural_network()
