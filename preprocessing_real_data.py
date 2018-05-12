import json, HTMLParser, string, re, itertools, nltk
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
printable = set(string.printable)

result = []

slang_replacer = CsvWordReplacer('dictionary_based/slang1.csv')
#file_object  = open("data/real_world_data.json", "r")
file_object  = open("Twitter_Program/data/basketball.json", "r")

for line in file_object: 
	
	try:
		if line in ['\n', '\r\n']:
			continue
		text = json.loads(line)['text']	
		lang = json.loads(line)['lang']
		#print text
		
		if lang == 'en':
			print "---------"
			# removing all non ascii characters
			text = filter(lambda x: x in printable, text)
			#print tweet

			a = re.sub(r"http\S+", "", text)   # Removing URLs
			a = html_parser.unescape(a)					   # Removing HTML characters
			a = RegexpReplacer().replace(a) 
			a = a.lower()
			a = str(a).translate(None, string.punctuation)               
			a = re.sub(r'\b\w{1,2}\b', '', a)            #remove word whose length is less than 3
		
			#print a

			word_tokens = word_tokenize(a)                      # Tokenization
		
			#print word_tokens
		
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


			#p = " ".join(p)
			p = " ".join(w for w in p if w in words)
			a = word_tokenize(p)
			length = len(a)

			print p
			result.append(p)
			print ""


			if length >= 3:
				print "-------------------------------"
				d = {'tweet': [p]}
				df = pd.DataFrame(data=d)

				#print df
	
				with open('data/Cleaned_data10.csv', 'a') as f:
					df.to_csv(f, header=False)
		
	except:
		continue