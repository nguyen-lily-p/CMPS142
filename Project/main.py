# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, csv, pandas, sys, string, numpy, compare_models, sklearn.metrics, performance_metrics
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# import scikit-learn functions for classifiers
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#nltk.download()

# default file path for training data
TRAINING_DATA_PATH = "train.csv"
# default file path for testing data
TESTING_DATA_PATH = "test.csv"
# default file path for output of predictions
OUTPUT_PATH = "output.csv"
# file path for output of performance metrics
OUTPUT_PERFORMANCE_PATH = "output_performance.txt"

# declares classifies that will be trained and used for testing
# global so all functions can access
naiveBayesModel = MultinomialNB()
linearSVCModel = LinearSVC()
logRegModel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 1)

# initializes array with previously declared classifiers to make voting simpler


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


    #compare_models.compare_models(train_data_df)
    #### preprocessing & feature extraction ####
    tfidf = TfidfVectorizer(tokenizer = tokenize)
    train_feature_set = tfidf.fit_transform(train_data_df["Phrase"])
    test_feature_set = tfidf.transform(test_data_df["Phrase"])

    # training - send to different algorithms
    model = trainClassifiers(train_feature_set, train_data_df["Sentiment"].tolist())
    print("\nInstances x Features): ", train_feature_set.shape)
    print("\nInstances x Features): ", test_feature_set.shape)


    # test
    predictions_df = pandas.DataFrame(model.predict(test_feature_set)) ### REPLACE WITH ACTUAL TEST SET BEFORE SUBMISSION
    predictions_df = pandas.concat([test_data_df["PhraseId"], predictions_df], axis = 1)
    print("Accuracy: ", model.score(test_feature_set, test_data_df["Sentiment"].tolist()))# REMOVE BEFORE SUBMISSION
    performance_metrics.get_performance(model, test_feature_set, test_data_df["Sentiment"].tolist(), args.perfFile)

    predictions_df.to_csv(path_or_buf = args.outFile, header = ["PhraseId", "Sentiment"], index = False)
    

if __name__ == '__main__':
    main()
