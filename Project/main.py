# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, csv, pandas, sys, string, numpy, compare_models
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# import scikit-learn functions for classifiers
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy import sparse


#nltk.download()

# default file path for training data
TRAINING_DATA_PATH = "train.csv"
# default file path for testing data
TESTING_DATA_PATH = "test.csv"
# default file path for output
OUTPUT_PATH = "output.csv"

NEGATIVE_WORDS_PATH = "negative-words.txt"
POSITIVE_WORDS_PATH = "positive-words.txt"

# declares classifies that will be trained and used for testing
# global so all functions can access
naiveBayesModel = MultinomialNB()
linearSVCModel = LinearSVC()
logRegModel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 1, max_iter=10000)

# initializes array with previously declared classifiers to make voting simpler

# trains multiple classifiers with training set, returns accuracy of each algorithm
# parameter is matrix of occurences of keywords in each phrase
def trainClassifiers(features, labels):
    # trains each classifier on given training set
    classArr = VotingClassifier(estimators = [('NB', naiveBayesModel), ('linSVC', linearSVCModel), ('LR', logRegModel)], \
            voting = 'hard')
    
    classArr = classArr.fit(features, labels)
    print(classArr.score(features, labels))
    
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
    # phrase = [PorterStemmer().stem(token.translate(mapping)) for token in phrase \
    #         if token.translate(mapping).isalpha()]

    phrase = [token.translate(mapping) for token in phrase \
            if token.translate(mapping).isalpha()]

    return phrase

def get_liwc_features(data):
    pass

def get_pos_features(data):
    pass

def get_unigram_bow_features(data):
    vectorizer = CountVectorizer() 
    matrix = vectorizer.fit_transform(data["Phrase"])
    return sparse.csr_matrix(matrix)

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

def get_idf_features(data):
    tfidf = TfidfVectorizer(tokenizer = tokenize)
    return tfidf.fit_transform(data["Phrase"])

def get_all_features(data):
    wc_matrix = get_word_count_features(data)
    uni_bow_matrix = get_unigram_bow_features(data)
    idf_matrix = get_idf_features(data)
    return sparse.hstack([idf_matrix, wc_matrix, uni_bow_matrix])

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

    #### feature extraction ####'
    train_feature_set = get_all_features(train_data_df)

    #compare_models.compare_models(train_data_df)
    #### preprocessing & feature extraction ####

    # switch to test df when available
    test_feature_set = get_all_features(train_data_df)



    # join matrices together

    # training - send to different algorithms
    
    model = trainClassifiers(train_feature_set, train_data_df["Sentiment"].tolist())

    # test
    predictions_df = pandas.DataFrame(model.predict(test_feature_set)) ### REPLACE WITH ACTUAL TEST SET BEFORE SUBMISSION
    print(predictions_df)
    predictions_df = pandas.concat([train_data_df["PhraseId"], predictions_df], axis = 1)
    print("Accuracy: ", model.score(test_feature_set, train_data_df["Sentiment"].tolist()))# REMOVE BEFORE SUBMISSION
    
    predictions_df.to_csv(path_or_buf = args.outFile, header = ["PhraseId", "Sentiment"], index = False)
    

if __name__ == '__main__':
    main()
