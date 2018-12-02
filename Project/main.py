# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, pandas, sys, string, numpy
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
logRegModel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 1)

# initializes array with previously declared classifiers to make voting simpler

    # remove punctuation, remove non-alphabetic tokens, stem tokens
# for i in range(0, phrase_df.size):
#     phrase_df[i] = [nltk.stem.PorterStemmer().stem(token.translate(mapping)) \
#             for token in phrase_df[i] if token.translate(mapping).isalpha()]

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
    # phrase = [PorterStemmer().stem(token.translate(mapping)) for token in phrase \
    #         if token.translate(mapping).isalpha()]

    phrase = [token.translate(mapping) for token in phrase \
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


    #### preprocessing ####
    #train_data_df["Phrase"] = preprocess(train_data_df["Phrase"])

    # remove training instances with empty phrase list after preprocessing
    #index = 0
    #while index < len(train_data_df.index):
    #    if not train_data_df["Phrase"].iloc[index]:
    #        train_data_df = train_data_df.drop(train_data_df.index[index])

    #    index += 1
    #print(train_data_df["Phrase"])


    #### feature extraction ####
    def get_liwc_features(train_data_df):
        pass
    
    def get_pos_features(train_data_df):
        pass

    def get_unigram_bow_features(train_data_df):
        vectorizer = CountVectorizer() 
        matrix = vectorizer.fit_transform(train_data_df["Phrase"])
        # print(vectorizer.get_feature_names())
        # print(feature)
        return sparse.csr_matrix(matrix)

    # returns a dense matrix of features 
    def get_word_count_features(train_data_df):
        negative_words = []
        with open(NEGATIVE_WORDS_PATH) as file:
            lines = file.readlines()
            negative_words = [line.strip() for line in lines]
        
        positive_words = []
        with open(POSITIVE_WORDS_PATH) as file:
            lines = file.readlines()
            positive_words = [line.strip() for line in lines] 

        # print(positive_words)
        dense_matrix = []
        for phrase in train_data_df["Phrase"]:
            word_array = tokenize(phrase)
            # print(word_array)
            # count number of positive and negative words in each phrase 
            num_pos = 0
            num_neg = 0
            for word in word_array:
                if(word in positive_words):
                    num_pos += 1
                if(word in negative_words):
                    num_neg +=  1
            # print("pos " + str(num_pos))
            # print("neg " + str(num_neg))
            # add them as tuples to feature vector
            dense_matrix.append([num_pos, num_neg])
        sparse_matrix = sparse.csr_matrix(dense_matrix)
        # print(sparse_matrix)
        return sparse_matrix

    wc_matrix = get_word_count_features(train_data_df)
    uni_bow_matrix = get_unigram_bow_features(train_data_df)
    print(uni_bow_matrix)
    
    tfidf = TfidfVectorizer(tokenizer = tokenize, min_df = 1)
    tfs = tfidf.fit_transform(train_data_df["Phrase"])
    
    # # print nice version of sparse matrix
    # print("\nDOCUMENT-TFIDF SPARSE MATRIX")
    # print(tfs)
    # feature_names = tfidf.get_feature_names()

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
    feature_matrix = sparse.hstack([tfs, wc_matrix, uni_bow_matrix])
    print(feature_matrix)
    trainClassifiers(feature_matrix, train_data_df["Sentiment"].tolist())

    # test


if __name__ == '__main__':
    main()
