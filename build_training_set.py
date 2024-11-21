import nltk
from simpleDemo import extract_features
from classifier_helper import get_word_features
import pandas as pd

# Load tweets from the local CSV file
tweets_df = pd.read_csv("data/Twitter_Data.csv")  # Adjust path if necessary

# Assuming the CSV has columns 'text' for tweet content and 'label' for sentiment
tweetItems = []
for index, row in tweets_df.iterrows():
    processed_tweet = row['text']
    opinion = row['label']  # Assuming there's a 'label' column for sentiment
    if opinion not in ['neutral', 'negative', 'positive']:
        print(f'Error with tweet = {processed_tweet}, Line = {index + 1}')
    tweet_item = (processed_tweet, opinion)
    tweetItems.append(tweet_item)

tweets = []    
for (words, sentiment) in tweetItems:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

def get_words_in_tweets(tweets):
    words = []
    for tweet in tweets:
        words.extend(tweet[0])
    return words

word_features = get_word_features(get_words_in_tweets(tweets))
nltk.classify.set_word_features(word_features)
training_set = nltk.classify.apply_features(extract_features, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)
tweet = 'I am so sad'
print(classifier.classify(extract_features(tweet.split())))
print(nltk.classify.accuracy(classifier, training_set))
classifier.show_most_informative_features(20)