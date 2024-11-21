import sys
import pandas as pd  # Import pandas for data handling
import baseline_classifier, naive_bayes_classifier, max_entropy_classifier, libsvm_classifier

keyword = 'iphone'
time = 'today'

# Load tweets from the local CSV file
tweets = pd.read_csv('data/Twitter_Data.csv')  # Adjust the path if necessary

# Check if an algorithm argument was provided
if len(sys.argv) < 2:
    print("Please choose the algorithm to test, syntax = python bass_analyze.py (svm|naivebayes|maxent|baseline)")
    exit()

algorithm = sys.argv[1]

if algorithm == 'baseline':
    bc = baseline_classifier.BaselineClassifier(tweets, keyword, time)
    bc.classify()
    val = bc.getHTML()

elif algorithm == 'naivebayes':
    trainingDataFile = 'data/full_training_dataset.csv'  # Ensure this file exists
    classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
    
    print('Started to instantiate Naive Bayes Classifier')
    nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time,
                                  trainingDataFile, classifierDumpFile, trainingRequired=True)
    
    nb.classify()  # Uncomment to classify tweets
    print('Naive Bayes Accuracy:', nb.accuracy())

elif algorithm == 'maxent':
    trainingDataFile = 'data/full_training_dataset.csv'  # Ensure this file exists
    classifierDumpFile = 'data/test/maxent_test_model.pickle'
    
    print('Started to instantiate Max Entropy Classifier')
    maxent = max_entropy_classifier.MaxEntClassifier(tweets, keyword, time,
                                  trainingDataFile, classifierDumpFile, trainingRequired=True)
    
    maxent.classify()  # Uncomment to classify tweets
    print('Max Entropy Accuracy:', maxent.accuracy())

elif algorithm == 'svm':
    trainingDataFile = 'data/full_training_dataset.csv'  # Ensure this file exists
    classifierDumpFile = 'data/test/svm_test_model.pickle'
    
    print('Started to instantiate SVM Classifier')
    sc = libsvm_classifier.SVMClassifier(tweets, keyword, time,
                                  trainingDataFile, classifierDumpFile, trainingRequired=True)
    
    sc.classify()  # Uncomment to classify tweets
    print('SVM Accuracy:', sc.accuracy())

print('Done')