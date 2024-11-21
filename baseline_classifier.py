import sys
import re
import classifier_helper, html_helper, pickle
import importlib  # Import importlib for reloading modules

# Reload sys to set default encoding (if needed, but usually not recommended)
importlib.reload(sys)
# Note: Setting default encoding is not a common practice in Python 3. 
# It may be better to handle string encoding explicitly where needed.

class BaselineClassifier:
    """ Classifier using baseline method """
    
    def __init__(self, data, keyword, time):
        # Instantiate classifier helper        
        self.helper = classifier_helper.ClassifierHelper('data/feature_list.txt')
        
        # Remove duplicates
        self.lenTweets = len(data)
        self.origTweets = self.getUniqData(data)
        self.tweets = self.getProcessedTweets(self.origTweets)
        
        self.results = {}
        self.neut_count = [0] * self.lenTweets
        self.pos_count = [0] * self.lenTweets
        self.neg_count = [0] * self.lenTweets

        self.time = time
        self.keyword = keyword
        self.html = html_helper.HTMLHelper()
    
    def getUniqData(self, data):
        uniq_data = {}        
        for i in data:
            d = data[i]
            u = []
            for element in d:
                if element not in u:
                    u.append(element)
            uniq_data[i] = u            
        return uniq_data
    
    def getProcessedTweets(self, data):        
        tweets = {}        
        for i in data:
            d = data[i]
            tw = []
            for t in d:
                tw.append(self.helper.process_tweet(t))
            tweets[i] = tw            
        return tweets
    
    def classify(self):
        # Load positive keywords file          
        with open("data/pos_mod.txt", "r") as inpfile:
            positive_words = [line.strip() for line in inpfile.readlines()]
            
        # Load negative keywords file    
        with open("data/neg_mod.txt", "r") as inpfile:
            negative_words = [line.strip() for line in inpfile.readlines()]
        
        # Process each tweet        
        for i in self.tweets:
            tw = self.tweets[i]
            count = 0
            res = {}
            for t in tw:
                neg_words = [word for word in negative_words if self.string_found(word, t)]
                pos_words = [word for word in positive_words if self.string_found(word, t)]
                
                if len(pos_words) > len(neg_words):
                    label = 'positive'
                    self.pos_count[i] += 1
                elif len(pos_words) < len(neg_words):
                    label = 'negative'
                    self.neg_count[i] += 1
                else:
                    if len(pos_words) > 0 and len(neg_words) > 0:
                        label = 'positive'
                        self.pos_count[i] += 1
                    else:
                        label = 'neutral'
                        self.neut_count[i] += 1
                
                result = {'text': t, 'tweet': self.origTweets[i][count], 'label': label}
                res[count] = result                
                count += 1
            
            self.results[i] = res
        
        filename = 'data/results_lastweek.pickle'
        with open(filename, 'wb') as outfile:        
            pickle.dump(self.results, outfile)

    def string_found(self, string1, string2):
        return re.search(r"\b" + re.escape(string1) + r"\b", string2) is not None
    
    def writeOutput(self, filename, writeOption='w'):
        with open(filename, writeOption) as fp:
            for i in self.results:
                res = self.results[i]
                for j in res:
                    item = res[j]
                    text = item['text'].strip()
                    label = item['label']
                    writeStr = f"{text} | {label}\n"
                    fp.write(writeStr)
    
    def getHTML(self):
        return self.html.getResultHTML(self.keyword, self.results, self.time, 
                                       self.pos_count, self.neg_count, 
                                       self.neut_count, 'baseline')