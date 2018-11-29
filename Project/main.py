# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, pandas, sys, nltk, string
#nltk.download()

# default file path for training data
TRAINING_DATA_PATH = "train.csv"
# default file path for testing data
TESTING_DATA_PATH = "test.csv"
# default file path for output
OUTPUT_PATH = "output.csv"

# import scikit-learn functions
from sklearn.naive_bayes import MultinomialNB

# import nltk functions
import nltk.classify.naivebayes

# trains multiple classifiers with training set, returns accuracy of each algorithm
def trainClassifiers(trainingSet):
    # trains NB classifier on given training set, prints accuracy
    naiveBayesClassifier = NaiveBayesClassifier.train(trainingSet)
    accuracy = nltk.classify.util.accuracy(classifier, trainingSet)
    print("Training Accuracy of Naive Bayes: " + (accuracy * 100))
    
def preprocess(phrase_df):
    phrase_df = phrase_df.str.lower() # convert strings to lowercase
    phrase_df = phrase_df.str.strip() # remove leading/trailing whitespace
    phrase_df = phrase_df.str.split(' ') # tokenize into words
    
    mapping = str.maketrans('', '', string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    
    for i in range(0, phrase_df.size ):
        # remove punctuation from tokens
        phrase_df[i] = [token.translate(mapping) for token in phrase_df[i]]
        # remove non-alphabetic tokens
        phrase_df[i] = [token for token in phrase_df[i] if token.isalpha()]
        # remove stop-words
        phrase_df[i] = [token for token in phrase_df[i] \
                if not token in stop_words]
        # stem the words
        phrase_df[i] = [nltk.stem.porter.PorterStemmer().stem(token) \
                for token in phrase_df[i]]

    return phrase_df

def main():
    # read in command-line arguments, if any
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


    # read training data into a pandas dataframe
    try:
        train_data_df = pandas.read_csv(args.trainFile)
    except FileNotFoundError:
        print("Error: File does not exist. File must be of type csv")
        sys.exit(1)
    except:
        print("Error: Unknown error occurred trying to read train data file")
        sys.exit(1)


    # preprocessing
    print("\n***PHRASE DATA BEFORE PREPROCESSING***")
    print(train_data_df["Phrase"])

    train_data_df["Phrase"] = preprocess(train_data_df["Phrase"])

    # remove instances with empty phrase list after preprocessing
    index = 0
    while index < len(train_data_df.index):
        if not train_data_df["Phrase"].iloc[index]:
            train_data_df = train_data_df.drop(train_data_df.index[index])

        index += 1

    print("\n***PHRASE DATA AFTER PREPROCESSING***")
    print(train_data_df["Phrase"])


    # feature extraction


    # training - send to different algorithms


    # test

if __name__ == '__main__':
    main()
