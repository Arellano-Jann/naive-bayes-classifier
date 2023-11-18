# Import libraries
import numpy as np
import pandas as pd
import argparse
import re
from unicodedata import normalize

class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.vocabulary = None
        self.training_set= None
        self.test_set = None
        self.p_spam = None
        self.p_ham = None
        self.test_set_path = test_set_path
        
        self.probability_word_spam = None # given probability that a word is spam
        self.probability_word_ham = None

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')

    # Normalization
    # Replace addresses (hhtp, email), numbers (plain, phone), money symbols
    def preprocess(self, text):
        text = text.lower() 
        text = re.sub(r'\S+@\S+|http\S+|[^\w\s]|\d+', '', text) # email addresses, URL (http), symbols, digits removal
        # text = re.sub(r'[^a-zA-Z]', '', text) # Remove anything that is not a letter # might need to manually remove addresses and url's tho
        
        text = normalize('NFKD', text) # strip unicode
        return text
    
    # Function for data preprocessing, including Normalization, lemmazation and stemming
    def data_cleaning(self): # output: spam/ham dataframe, vocab/dictionary, vocab - all unique words in the training set
        from sklearn.feature_extraction.text import CountVectorizer # tokenizes, removes stop words,
        count_vector = CountVectorizer(preprocessor=self.preprocess, stop_words='english')
        # count_vector = CountVectorizer(stop_words='english', strip_accents='unicode')
        vectorized_data = count_vector.fit_transform(self.training_set.v2)
        words = count_vector.get_feature_names_out()
        df = pd.DataFrame(vectorized_data.toarray(), columns=words)
        self.training_set = pd.concat([self.training_set, df], axis=1) # we concat here so that we only have a single df that we can call over
        # print('df\n',self.training_set)
        
        # Create vocab - list of all unique words
        self.vocabulary = words


    # Function for Naive Bayes algorithm
    def fit_bayes(self): # output: p_spam, p_ham, - probability that a word is spam or ham. 
        # Separate the spam and ham dataframes
        spam_df = self.training_set[self.training_set['v1'] == 'spam']
        ham_df = self.training_set[self.training_set['v1'] == 'ham']
        
        # Calculate P(Spam) and P(Ham)
        # the probability of spam is the amount of spam rows over total rows
        # this is basically the base probabilty that a message may be spam - prior probabilty
        self.p_spam = len(spam_df) / len(self.training_set)
        self.p_ham = len(ham_df) / len(self.training_set)
        
        # Calculate Nspam, Nham and Nvocabulary
        # number of spam/ham words in all the messages
        def calc_words(df):
            for row in df['v2']:
                yield len(row.split())
        n_spam_words = sum(calc_words(spam_df))
        n_ham_words = sum(calc_words(ham_df))
        n_vocab = len(self.vocabulary)
        # print('p_spam', p_spam, '\np_ham', p_ham, '\nn_spam_words', (n_spam_words), '\nn_ham_words', (n_ham_words), '\nn_vocab', (n_vocab))
        
        # Laplace smoothing parameter
        alpha = 1

        # Calculate P(wi|Spam) and P(wi|Ham)
        # calculate the probability that a word is a spam or ham
        self.probability_word_spam = {word:0 for word in self.vocabulary} # given probability that a word is spam
        self.probability_word_ham = {word:0 for word in self.vocabulary} # this will be used to calculate whether a msg is spam
        for word in self.vocabulary:
            n_word_given_spam = spam_df[word].sum() # this will take the sum of a single word in all spam msgs # basically, the number of times it appears in all spams
            self.probability_word_spam[word] = (n_word_given_spam + alpha) / (n_spam_words + alpha * n_vocab) # this will calculate the probabilty that a word is actually spam
            
            n_word_given_ham = ham_df[word].sum() # this will take the sum of a single word in all ham msgs # basically, the number of times it appears in all hams
            self.probability_word_ham[word] = (n_word_given_ham + alpha) / (n_ham_words + alpha * n_vocab) # this will calculate the probabilty that a word is actually ham
        
            

    # Trains the cleaned data to get a Naive Bayes model
    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()
    
    # Function to classify a single SMS text as spam or ham
    def sms_classify(self, message): # output: none - whether a message is ham or spam. do naive_bayes/multiply out all words here
        '''
        classifies a single message as spam or ham
        Takes in as input a new sms (w1, w2, ..., wn),
        performs the same data cleaning steps as in the training set,
        calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
        compares them and outcomes whether the message is spam or not.
        '''
        message = self.preprocess(message).split()
        
        def fun(msg_type):
            for word in message:
                if msg_type == 'spam': yield self.probability_word_spam.get(word, 1) # if it's not in spam wordlist, multiply by 1 to ensure data validation
                if msg_type == 'ham': yield self.probability_word_ham.get(word, 1)
            
        from math import prod
        p_ham_given_message = self.p_ham * prod(fun('ham'))
        p_spam_given_message = self.p_spam * prod(fun('spam'))
        
        # print('ham probability', p_ham_given_message)
        # print('spam probability', p_spam_given_message)

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_spam_given_message > p_ham_given_message:
            return 'spam'
        else:
            return 'needs human classification'

    # Function to classify all the texts in the validation or test dataset
    def classify_test(self):
        '''
        Calculate the accuracy of the algorithm on the test set and returns 
        the accuracy as a percentage.
        '''
        # need to do bayes on each 
        accuracy = 0
        self.train()
        # self.sms_classify('08714712388 between 10am-7pm Cost 10p')
        # print(self.test_set)
        for index, row in self.test_set.iterrows():
            # print(row['v1'], row['v2'])
            classification = self.sms_classify(row['v2'])
            if classification == row['v1']: # then accurate
                accuracy += 1
        return accuracy / len(self.test_set) * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
