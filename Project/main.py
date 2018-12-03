# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, csv, pandas, sys, string, numpy, compare_models, sklearn.metrics, performance_metrics
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# import scikit-learn functions for classifiers
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import sparse

#nltk.download()

# default file path for training data
TRAINING_DATA_PATH = "train.csv"
# default file path for testing data
TESTING_DATA_PATH = "test.csv"
# default file path for output of predictions
OUTPUT_PATH = "output.csv"
# default file path for output of performance metrics
OUTPUT_PERFORMANCE_PATH = "output_performance.txt"

NEGATIVE_WORDS_PATH = "negative-words.txt"
POSITIVE_WORDS_PATH = "positive-words.txt"

# declares classifies that will be trained and used for testing
# global so all functions can access
naiveBayesModel = MultinomialNB()
linearSVCModel = LinearSVC()
logRegModel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 1, max_iter=10000)

# trains multiple classifiers with training set, returns accuracy of each algorithm
# parameter is matrix of occurences of keywords in each phrase
def trainClassifiers(features, labels):
    """
        Trains multiple classifiers with training set
        Parameters are the features set and the labels of the instances
        Returns the trained model
    """
    # trains each classifier on given training set
    classArr = VotingClassifier(estimators = [('NB', naiveBayesModel), ('linSVC', linearSVCModel), ('LR', logRegModel)], \
            voting = 'hard')
    
    classArr = classArr.fit(features, labels)
    
    return classArr

    
def tokenize(phrase_str):
    """
        Performs tokenization and some preprocessing operations on text data.
        Converts a phrase into a list of words, removes punctuation, removes
            non-alphabetic tokens, and stems the tokens
        Returns the list of tokens
    """
    phrase = phrase_str.split(' ') # tokenize string by space character

    mapping = str.maketrans('', '', string.punctuation)

    # remove punctuation, remove non-alphabetic tokens, stem tokens
    phrase = [PorterStemmer().stem(token.translate(mapping)) for token in phrase \
             if token.translate(mapping).isalpha()]

    #phrase = [token.translate(mapping) for token in phrase \
    #        if token.translate(mapping).isalpha()]

    return phrase

"""
def get_liwc_features(data):
    pass

def get_pos_features(data):
    pass
"""

def get_unigram_bow_features(train_data, test_data):
    vectorizer = CountVectorizer() 
    vectorizer = vectorizer.fit(train_data)
    return vectorizer.transform(train_data), vectorizer.transform(test_data)

"""
# returns a dense matrix of features 
def get_word_count_features(data):
    negative_words = []
    with open(NEGATIVE_WORDS_PATH) as file:
        lines = file.readlines()
        negative_words = [line.strip() for line in lines]
    
    positive_words = []
    with open(POSITIVE_WORDS_PATH) as file:
        lines = file.readlines()
        positive_words = [line.strip() for line in lines] 

    dense_matrix = []
    for phrase in data["Phrase"]:
        word_array = tokenize(phrase)
        # count number of positive and negative words in each phrase 
        num_pos = 0
        num_neg = 0
        for word in word_array:
            if(word in positive_words):
                num_pos += 1
            if(word in negative_words):
                num_neg +=  1
        dense_matrix.append([num_pos, num_neg])
    return sparse.csr_matrix(dense_matrix)
"""

def get_idf_features(train_data, test_data):
    tfidf = TfidfVectorizer(tokenizer = tokenize)
    tfidf.fit(train_data)
    return tfidf.transform(train_data), tfidf.transform(test_data)

def get_all_features(train_data, test_data):
    #train_wc_matrix, test_wc_matrix = get_word_count_features(train_data, test_data)
    train_uni_bow_matrix, test_uni_bow_matrix = get_unigram_bow_features(train_data, test_data)
    train_idf_matrix, test_idf_matrix = get_idf_features(train_data, test_data)
    #return sparse.hstack([train_idf_matrix, train_wc_matrix, train_uni_bow_matrix]), \
    #       sparse.hstack([test_idf_matrix, test_wc_matrix, test_uni_bow_matrix])
    return sparse.hstack([train_idf_matrix, train_uni_bow_matrix]), \
           sparse.hstack([test_idf_matrix, test_uni_bow_matrix])


def main():
    #### read in command-line arguments, if any ####
    parser = argparse.ArgumentParser(description = "program to predict the "
            "sentiment of a given phrase")
    parser.add_argument("--train", dest = "trainFile", \
            default = TRAINING_DATA_PATH, type = str, \
            help = "the path to the .csv file containing the training data")
    parser.add_argument("--test", dest = "testFile", \
            default = TESTING_DATA_PATH, type = str, \
            help = "the path to the .csv file containing the test data")
    parser.add_argument("--out", dest = "outFile", default = OUTPUT_PATH, \
            type = str, help = "the path of the output file")
    parser.add_argument("--perf", dest = "perfFile", default = \
            OUTPUT_PERFORMANCE_PATH, type = str, help = "the path of the performance "
            "output file")
    args = parser.parse_args()

    #### read training and testing data into a pandas dataframe ####
    try:
        train_data_df = pandas.read_csv(args.trainFile)
        test_data_df = pandas.read_csv(args.testFile)
    except FileNotFoundError:
        print("Error: File does not exist. File must be of type csv")
        sys.exit(1)
    except:
        print("Error: Unknown error occurred trying to read train data file")
        sys.exit(1)

    #### preprocessing & feature extraction ####
    train_feature_set, test_feature_set = get_all_features(train_data_df["Phrase"], test_data_df["Phrase"])

    #compare_models.compare_models(train_data_df)

    ### training ###   
    model = trainClassifiers(train_feature_set, train_data_df["Sentiment"].tolist())

    ### test ###
    predictions_df = pandas.DataFrame(model.predict(test_feature_set))
    predictions_df = pandas.concat([test_data_df["PhraseId"], predictions_df], axis = 1)
    predictions_df.to_csv(path_or_buf = args.outFile, header = ["PhraseId", "Sentiment"], index = False)
    
    # write performance stats to txt file
    perf_out_file = open(args.perfFile, "w")
    #performance_metrics.get_performance_train(model, train_feature_set, train_data_df["Sentiment"].tolist(), perf_out_file, True)
    performance_metrics.get_performance_cv(model, train_feature_set, train_data_df["Sentiment"].tolist(), perf_out_file, 3)
    perf_out_file.close()

if __name__ == '__main__':
    main()
