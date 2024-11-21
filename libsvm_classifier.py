from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from classifier_helper import ClassifierHelper
from html_helper import HTMLHelper

# Start Class
class SVMClassifier:
  """ SVM Classifier """

  # Variables
  helper = None
  classifier = None
  len_tweets = 0
  orig_tweets = None
  tweets = None
  results = None
  training_data_file = None

  # Init
  def __init__(self, data, keyword, time, training_data_file, classifier_dump_file, training_required=0):
    self.helper = ClassifierHelper('data/feature_list.txt')
    self.len_tweets = len(data)
    self.orig_tweets = self.get_unique_data(data)
    self.tweets = self.get_processed_tweets(self.orig_tweets)
    self.results = {}
    self.training_data_file = training_data_file
    self.time = time
    self.keyword = keyword

    # Train model if required
    if training_required:
      self.classifier = self.get_svm_trained_classifier(training_data_file, classifier_dump_file)
    else:
      try:
        # Load classifier from dump file
        self.classifier = self.load_model(classifier_dump_file)
      except FileNotFoundError:
        # Train if file not found
        self.classifier = self.get_svm_trained_classifier(training_data_file, classifier_dump_file)

  # Get unique data
  def get_unique_data(self, data):
    unique_data = {}
    for i, d in data.items():
      unique_data[i] = list(set(d))  # Use set to remove duplicates
    return unique_data

  # Get processed tweets
  def get_processed_tweets(self, data):
    tweets = {}
    for i, d in data.items():
      tw = []
      for t in d:
        tw.append(self.helper.process_tweet(t))
      tweets[i] = tw
    return tweets

  # Train SVM classifier
  def get_svm_trained_classifier(self, training_data_file, classifier_dump_file):
    # Read training data and labels
    tweet_items = self.get_filtered_training_data(training_data_file)

    tweets, labels = [], []
    for words, sentiment in tweet_items:
      words_filtered = [e.lower() for e in words.split() if self.helper.is_ascii(e)]
      tweets.append(words_filtered)
      labels.append(sentiment)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(tweets)

    # Train SVM model using SVC
    self.classifier = SVC(kernel='linear')  # Set kernel to linear
    self.classifier.fit(feature_vectors, labels)

    # Save model (optional)
    self.save_model(classifier_dump_file)

    return self.classifier

  # Load model from dump file (assuming pickle format)
  def load_model(self, model_file):
    with open(model_file, 'rb') as f:
      return pickle.load(f)

  # Save model to dump file (assuming pickle format)
  def save_model(self, model_file):
    with open(model_file, 'wb') as f:
      pickle.dump(self.classifier, f)

  # Get filtered training data (similar logic as before)
  def get_filtered_training_data(self, training_data_file):
    # ... (implement similar logic to filter data based on sentiment count)

    return tweet_items

  # Classify tweets
  def classify(self):
    for i, tw in self.tweets.items():
      test_tweets = []
      res = {}
      for words in tw:
        words_filtered = [e.lower() for e in words.split() if self.helper.is_ascii(e)]
        test_tweets.append(words_filtered)

      # Convert test tweets to feature vectors using the same vectorizer
      test_feature_vectors = vectorizer.transform(test_tweets)

      # Predict labels using the trained classifier
      p_labels = self.classifier.predict(test_feature_vectors)

      count = 0
      for t in tw:
        label = p_labels[count]
        if label == 0:
          label = 'positive'
          self.pos_count[i] += 1
        elif label == 1:
          label = 'negative'
          self.neg_count[i] += 1
        elif label == 2:
          label = 'neutral'
          self.neut_count[i] += 1
        result = {'text': t, 'tweet': self.orig_tweets[i][count], 'label': label}
        res[count] = result
        count += 1

      self.results[i] = res

  # Write output
  def write_output(self, filename, write_option='w'):
    # ... (implement write output logic)
    fp = open(filename, writeOption)
    for i in self.results:
            res = self.results[i]
            for j in res:
                item = res[j]
                text = item['text'].strip()
                label = item['label']
                writeStr = text+" | "+label+"\n"
                fp.write(writeStr)
            #end inner loop
        #end outer loop      
    #end writeOutput 

  # Accuracy
  def accuracy(self):
    # ... (implement accuracy calculation logic)
    tweets = self.getFilteredTrainingData(self.trainingDataFile)
    test_tweets = []
    for (t, l) in tweets:
            words_filtered = [e.lower() for e in t.split() if(self.helper.is_ascii(e))]
            test_tweets.append(words_filtered)

    test_feature_vector = self.helper.getSVMFeatureVector(test_tweets)
    p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),\
                                            test_feature_vector, self.classifier)
    count = 0
    total , correct , wrong = 0, 0, 0
    self.accuracy = 0.0
    for (t,l) in tweets:
            label = p_labels[count]
            if(label == 0):
                label = 'positive'
            elif(label == 1):
                label = 'negative'
            elif(label == 2):
                label = 'neutral'

            if(label == l):
                correct+= 1
            else:
                wrong+= 1
            total += 1
            count += 1
        #end loop
    self.accuracy = (float(correct)/total)*100
    print('Total = %d, Correct = %d, Wrong = %d, Accuracy = %.2f' % (total, correct, wrong, self.accuracy))
    (total, correct, wrong, self.accuracy)        
    #end


  # Get HTML
  def get_html(self):
    return self.html.get_result_html(self.keyword, self.results, self.time, self.pos_count, \
                                     self.neg_count, self.neut_count, 'svm')