from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.corpus import twitter_samples, stopwords
import sys, random
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(all_instances, n=None):

    n = len(all_instances)
    train_set = all_instances[:int(.8*n)]
    test_set = all_instances[int(.8*n):n]

    return train_set, test_set






# Different customizations for the TweetTokenizer
tokenizer = TweetTokenizer(preserve_case=False)
# tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True)
# tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)


fields = ['id', 'text']
positive_json = twitter_samples.tokenized("positive_tweets.json")
positive_csv = 'positive_tweets.csv'
#json2csv_preprocess(positive_json, positive_csv, fields, limit=n_instances)

#print positive_json


negative_json = twitter_samples.tokenized("negative_tweets.json")
negative_csv = 'negative_tweets.csv'
#json2csv_preprocess(negative_json, negative_csv, fields, limit=n_instances)

#neg_docs = parse_tweets_set(negative_csv, label='neg', word_tokenizer=tokenizer, sent_tokenizer=sent_tokenize)
#pos_docs = parse_tweets_set(positive_csv, label='pos', word_tokenizer=tokenizer, sent_tokenizer=sent_tokenize)

pos_data = pd.DataFrame(columns=['tweet','senti'])
pos_data['tweet'] = positive_json
pos_data['senti'] = 'pos'

#print pos_data

neg_data = pd.DataFrame(columns=['tweet','senti'])
neg_data['tweet'] = negative_json
neg_data['senti'] = 'neg'

result = pd.concat([pos_data, neg_data])
result = result.sample(frac=1).reset_index(drop=True)

#print result


training_tweets, testing_tweets = split_train_test(result)
#x_train, x_test, y_train, y_test =  train_test_split(result['tweet'], result['senti'], test_size=0.20, random_state=0)

sentim_analyzer = SentimentAnalyzer()

stopwords = stopwords.words('english')
all_words = [word for word in sentim_analyzer.all_words(training_tweets) if word.lower() not in stopwords]

print(all_words)

# Add simple unigram word features
unigram_feats = sentim_analyzer.unigram_word_feats(all_words, top_n=1000)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# Add bigram collocation features
bigram_collocs_feats = sentim_analyzer.bigram_collocation_feats([tweet[0] for tweet in training_tweets],top_n=100, min_freq=12)
sentim_analyzer.add_feat_extractor(extract_bigram_feats, bigrams=bigram_collocs_feats)

training_set = sentim_analyzer.apply_features(training_tweets)
test_set = sentim_analyzer.apply_features(testing_tweets)

classifier = sentim_analyzer.train(trainer, training_set)
# classifier = sentim_analyzer.train(trainer, training_set, max_iter=4)
try:
    classifier.show_most_informative_features()
except AttributeError:
    print('Your classifier does not provide a show_most_informative_features() method.')
results = sentim_analyzer.evaluate(test_set)



"""

    sentim_analyzer = SentimentAnalyzer()
    # stopwords = stopwords.words('english')
    # all_words = [word for word in sentim_analyzer.all_words(training_tweets) if word.lower() not in stopwords]
    all_words = [word for word in sentim_analyzer.all_words(training_tweets)]

    # Add simple unigram word features
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words, top_n=1000)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

    # Add bigram collocation features
    bigram_collocs_feats = sentim_analyzer.bigram_collocation_feats([tweet[0] for tweet in training_tweets],
        top_n=100, min_freq=12)
    sentim_analyzer.add_feat_extractor(extract_bigram_feats, bigrams=bigram_collocs_feats)

    training_set = sentim_analyzer.apply_features(training_tweets)
    test_set = sentim_analyzer.apply_features(testing_tweets)

    classifier = sentim_analyzer.train(trainer, training_set)
    # classifier = sentim_analyzer.train(trainer, training_set, max_iter=4)
    try:
        classifier.show_most_informative_features()
    except AttributeError:
        print('Your classifier does not provide a show_most_informative_features() method.')
    results = sentim_analyzer.evaluate(test_set)

    if output:
        extr = [f.__name__ for f in sentim_analyzer.feat_extractors]
        output_markdown(output, Dataset='labeled_tweets', Classifier=type(classifier).__name__,
                        Tokenizer=tokenizer.__class__.__name__, Feats=extr,
                        Results=results, Instances=n_instances)
"""