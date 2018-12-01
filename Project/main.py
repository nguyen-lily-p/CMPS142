# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, pandas, sys, string, numpy
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# import scikit-learn functions for classifiers
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#nltk.download()

# default file path for training data
TRAINING_DATA_PATH = "train.csv"
# default file path for testing data
TESTING_DATA_PATH = "test.csv"
# default file path for output
OUTPUT_PATH = "output.csv"

# declares classifies that will be trained and used for testing
# global so all functions can access
naiveBayesModel = MultinomialNB()
linearSVCModel = LinearSVC()
logRegModel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 1)

# initializes array with previously declared classifiers to make voting simpler


# trains multiple classifiers with training set, returns accuracy of each algorithm
# parameter is matrix of occurences of keywords in each phrase
def trainClassifiers(features, labels):
    # trains each classifier on given training set
    classArr = VotingClassifier(estimators = [('NB', naiveBayesModel), ('linSVC', linearSVCModel), ('LR', logRegModel)], \
            voting = 'hard')
    
    classArr = classArr.fit(features, labels)
    
    #predictions = pandas.DataFrame({"Prediction": classArr.predict(features)})
    #predictions.to_csv(path_or_buf = OUTPUT_PATH, index = False)

    
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

    return phrase

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


    #### read training data into a pandas dataframe ####
    try:
        train_data_df = pandas.read_csv(args.trainFile)
    except FileNotFoundError:
        print("Error: File does not exist. File must be of type csv")
        sys.exit(1)
    except:
        print("Error: Unknown error occurred trying to read train data file")
        sys.exit(1)


    #### preprocessing & feature extraction ####
    tfidf = TfidfVectorizer(tokenizer = tokenize, min_df = 1)
    tfs = tfidf.fit_transform(train_data_df["Phrase"])
    

    ######## PRINTING MATRIX OF FEATURE SET -- REMOVE EVENTUALLY ###################
    # print nice version of sparse matrix
    #print("\nDOCUMENT-TFIDF SPARSE MATRIX")
    #print(tfs)
    #feature_names = tfidf.get_feature_names()

    # print word and tfidf score for first 100 documents
    #print("\nWORD AND TFIDF SCORE FOR FIRST 100 DOCUMENTS")
    #for idx in range(0, 100):
    #    feature_index = tfs[idx,:].nonzero()[1]
    #    tfidf_scores = zip(feature_index, [tfs[idx, x] for x in feature_index])

    #    print("\n*** DOCUMENT ", idx, " ***")
    #    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
    #        print(w, " --- ", s)

    # print (# of rows, # of columns) of matrix, i.e. (instances x features)
    #print("\n(INSTANCES, FEATURES): ", tfs.shape)
    ################################################################################


    # training - send to different algorithms
    trainClassifiers(tfs, train_data_df["Sentiment"].tolist())
    

    # test


if __name__ == '__main__':
    main()
