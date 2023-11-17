# Import libraries
import numpy as np
import pandas as pd
import argparse

class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.vocabulary = None
        self.training_set= None
        self.test_set = None
        self.p_spam = None
        self.p_ham = None
        self.test_set_path = test_set_path

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')

    # Function for data preprocessing, including Normalization, lemmazation and stemming
    def data_cleaning(self): # output: spam/ham dataframe, vocab/dictionary, 
        # Normalization
        # Replace addresses (hhtp, email), numbers (plain, phone), money symbols
        # Remove the stop-words

        # Lemmatization - Graduate Students

        # Stemming - Gradutate Students

        # Tokenization

        # Vectorization

        # Remove duplicates - Can you think of any data structure that can help you remove duplicates?

        # Create the dictionary
        
        # Convert to dataframe 

        # Separate the spam and ham dataframes
        pass

    # Function for Naive Bayes algorithm
    def fit_bayes(self): # output: p_spam, p_ham - probability that SMS messages that are spam or ham
        # Calculate P(Spam) and P(Ham)
        
        # Calculate Nspam, Nham and Nvocabulary
        
        # Laplace smoothing parameter
        alpha = 1

        # Calculate P(wi|Spam) and P(wi|Ham)

    # Trains the cleaned data to get a Naive Bayes model
    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()
    
    # Function to classify a single SMS text as spam or ham
    def sms_classify(self, message): # output: none. whether a message is ham or spam. maybe do fit_bayes here
        '''
        classifies a single message as spam or ham
        Takes in as input a new sms (w1, w2, ..., wn),
        performs the same data cleaning steps as in the training set,
        calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
        compares them and outcomes whether the message is spam or not.
        '''
        

        # if p_ham_given_message > p_spam_given_message:
        #     return 'ham'
        # elif p_spam_given_message > p_ham_given_message:
        #     return 'spam'
        # else:
        #     return 'needs human classification'
        pass

    # Function to classify all the texts in the validation or test dataset
    def classify_test(self):
        '''
        Calculate the accuracy of the algorithm on the test set and returns 
        the accuracy as a percentage.
        '''
        # need to do bayes on each 
        accuracy = 99
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
