#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import HTMLParser, string, re, itertools, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from autocorrect import spell
from textblob import TextBlob
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from replacers import *

replacer = RepeatReplacer()
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
html_parser = HTMLParser.HTMLParser()
words = set(nltk.corpus.words.words())
slang_replacer = CsvWordReplacer('dictionary_based/slang1.csv')

text = pd.read_csv('data/Training_data_without_neutral.csv')
#text = pd.read_csv('data/Training_data_with_neutral.csv')

text.drop(text.columns[0],axis=1,inplace=True)

output_file = 'data/Cleaned_data_without_neutral.csv'
#output_file = 'data/Cleaned_data_with_neutral.csv'

text_doc = open(output_file, 'w')
text_doc.write(',senti,tweet\n') 
text_doc.close()

print "----------BEFORE------------"
print text.head(),"\n"

result = []
printable = set(string.printable)

i=0
for index, row in text.iterrows():
	# removing all non ascii characters
	try:
		tweet = filter(lambda x: x in printable, row['tweet'])
		#print tweet

		a = re.sub(r"http\S+", "", tweet)   # Removing URLs
		a = html_parser.unescape(a)					   # Removing HTML characters
		a = RegexpReplacer().replace(a) 
		a = a.lower()            
		
		print a

		word_tokens = word_tokenize(a)                      # Tokenization
		stop_words = set(stopwords.words('english'))

		s = []
		for a in word_tokens:
			a = slang_replacer.replace(a)
			s.append(a)

		s = " ".join(s)
		s = str(s).translate(None, string.punctuation)   
		
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

		p = " ".join(w for w in p if w in words)
		p = re.sub(r'\b\w{1,2}\b', '', p)
		a = word_tokenize(p)
		length = len(a)

		p = " ".join(a)
		row['tweet'] = p
		
		if length >= 3:
			d = {'senti': [row['senti']], 'tweet': [row['tweet']]}
			df = pd.DataFrame(data=d)
			with open(output_file, 'a') as f:
				df.to_csv(f, header=False)
	except:
		continue	