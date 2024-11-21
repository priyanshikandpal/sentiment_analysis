import pandas as pd  # Import pandas for data handling
import baseline_classifier, naive_bayes_classifier, max_entropy_classifier, libsvm_classifier
import sys

keyword = 'iphone'
time = 'today'

# Load tweets from the local CSV file
tweets = pd.read_csv('data/Twitter_Data.csv')  # Adjust the path if necessary

if len(sys.argv) < 2:
    print("Please choose the algorithm to test, syntax = python analyze.py (svm|naivebayes|maxent|baseline)")
    sys.exit()

algorithm = sys.argv[1]

# Ensure valid algorithm selection
if algorithm not in ['baseline', 'naivebayes', 'maxent', 'svm']:
    print(f"Invalid algorithm '{algorithm}' selected. Choose from 'svm', 'naivebayes', 'maxent', or 'baseline'.")
    sys.exit()

# Classifier selection
if algorithm == 'baseline':
    bc = baseline_classifier.BaselineClassifier(tweets, keyword, time)
    bc.classify()
    val = bc.getHTML()

elif algorithm == 'naivebayes':
    trainingDataFile = 'data/training_trimmed.csv'  # Ensure this file exists
    classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
    
    nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time, trainingDataFile, classifierDumpFile, trainingRequired=True)
    nb.classify()
    print('Naive Bayes Accuracy:', nb.accuracy())

elif algorithm == 'maxent':
    trainingDataFile = 'data/full_training_dataset.csv'  # Ensure this file exists
    classifierDumpFile = 'data/test/maxent_test_model.pickle'
    
    maxent = max_entropy_classifier.MaxEntClassifier(tweets, keyword, time, trainingDataFile, classifierDumpFile, trainingRequired=True)
    maxent.classify()
    print('Max Entropy Accuracy:', maxent.accuracy())

elif algorithm == 'svm':
    trainingDataFile = 'data/training_trimmed.csv'  # Ensure this file exists
    classifierDumpFile = 'data/test/svm_test_model.pickle'
    
    sc = libsvm_classifier.SVMClassifier(tweets, keyword, time, trainingDataFile, classifierDumpFile, trainingRequired=True)
    sc.classify()
    print('SVM Accuracy:', sc.accuracy())