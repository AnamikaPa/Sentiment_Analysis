import numpy as np 
import pandas as pd 


df = pd.read_csv('data/finalizedfull.csv')
#print df.head()
sentiment = {0:-1,2:0,4:1}
df['senti'] = df['senti'].map(sentiment)
print df.head()


df1 = pd.read_csv('data/data.csv')
#print df1.head()

df1 = df1[['SentimentText','Sentiment']]
#print df1.head()

sentiment = {0:-1,1:1}
df1['Sentiment'] = df1['Sentiment'].map(sentiment)
#print df1.head()

df1.rename(columns = {'Sentiment':'senti','SentimentText':'tweet'}, inplace=True)
#print df1.head()

df2 = pd.read_csv('data/pos_sentiment.txt',header=None, names=['tweet'])
df2['senti'] =1
#print df2.head()

df3 = pd.read_csv('data/neg_sentiment.txt',header=None, names=['tweet'])
df3['senti'] =-1
#print df3.head()

sentiment = {0:-1,2:0,4:1}
df4 = pd.read_csv('data/main_train.csv', names = ["senti","b","c","d","e","tweet"])
df4.drop(["b","c","d","e"],axis=1,inplace=True)
df4['senti'] = df4['senti'].map(sentiment)
print df4.head()


df5 = pd.read_csv('data/main_test.csv', names = ["senti","b","c","d","e","tweet"])
df5.drop(["b","c","d","e"],axis=1,inplace=True)
df5['senti'] = df5['senti'].map(sentiment)
print df5.head()

result = pd.concat([df,df1,df2,df3,df4,df5])

#result = pd.concat([df,df5,df4])

print "\n\nNumber of Tweets in first data set: ",len(df)
print "Number of Tweets in second data set: ",len(df1)
print "Number of Tweets in third data set: ",len(df2)
print "Number of Tweets in fourth data set: ",len(df3)
print "Number of Tweets in fifth data set: ",len(df4)
print "Number of Tweets in sixth data set: ",len(df5) 

print "\n\nNumber of Tweets in Integrated data: ",len(result) ,"\n\n"

result = result.sample(frac=1).reset_index(drop=True)
print result.shape

result1 = result[result.senti != 0]
print result.shape
#df = result.groupby('senti').count()
#print df


integration = pd.DataFrame(result1).to_csv('data/Training_data_without_neutral.csv')
integration = pd.DataFrame(result).to_csv('data/Training_data_with_neutral.csv')