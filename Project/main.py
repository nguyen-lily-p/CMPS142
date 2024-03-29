# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, pandas, sys, string, numpy, sklearn.metrics, performance_metrics, word_category_counter
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import sparse

#nltk.download() # uncomment this line to automatically install NLTK library


# default file path for training data
TRAINING_DATA_PATH = "train.csv"
# default file path for testing data
TESTING_DATA_PATH = "testset_1.csv"
# default file path for output of predictions
OUTPUT_PATH = "output.csv"
# default file path for output of performance metrics
OUTPUT_PERFORMANCE_PATH = "output_performance.txt"


# declares classifies that will be trained and used for testing
# global so all functions can access
naiveBayesModel = MultinomialNB()
linearSVCModel = LinearSVC(penalty = 'l1', dual = False)
logRegModel = LogisticRegression(C = 1.5, solver = 'lbfgs', multi_class = 'multinomial', random_state = 1, max_iter=1000)

def trainClassifiers(features, labels):
    """
        Trains multiple classifiers with training set
        Parameters are the features set matrix and the labels of the instances
        Returns the trained model
    """
    # trains each classifier on given training set
    classArr = VotingClassifier(estimators = [('NB', naiveBayesModel), ('linSVC', linearSVCModel), ('LR', logRegModel)], \
            voting = 'hard', weights = [1, 5, 3])
    
    classArr = classArr.fit(features, labels)
    return classArr

    
def tokenize(phrase_str):
    """
        Performs tokenization and some preprocessing operations on text data.
        Converts a phrase into a list of words, removes punctuation, removes
            non-alphabetic tokens, and lemmatizes the tokens
        Returns the list of tokens
    """
    phrase = phrase_str.split(' ') # tokenize string by space character

    mapping = str.maketrans('', '', string.punctuation)

    # remove punctuation, remove non-alphabetic tokens, stem tokens
    phrase = [WordNetLemmatizer().lemmatize(token.translate(mapping)) for token in phrase \
             if token.translate(mapping).isalpha()]

    return phrase


liwc_categories =  [
'Total Pronouns', 'Total Function Words', 'Personal Pronouns', 'First Person Singular', 'First Person Plural', 
'Second Person', 'Third Person Singular', 'Third Person Plural', ' Impersonal Pronouns', 'Articles', 'Common Verbs',
'Auxiliary Verbs', 'Past Tense', 'Present Tense', 'Future Tense', 'Adverbs', 'Prepositions', 'Conjunctions',
'Negations', 'Quantifiers', 'Number', 'Swear Words', 'Social Processes', 'Family', 'Friends', 'Humans',
'Affective Processes', 'Positive Emotion', 'Negative Emotion', 'Anxiety', 'Anger', 'Sadness', 'Cognitive Processes',
'Insight', 'Causation', 'Discrepancy', 'Tentative', 'Certainty', 'Inhibition', 'Inclusive', 'Exclusive',
'Perceptual Processes', 'See', 'Hear', 'Feel', 'Biological Processes', 'Body', 'Health', 'Sexual', 'Ingestion',
'Relativity', 'Motion', 'Space', 'Time', 'Work', 'Achievement', 'Leisure', 'Home', 'Money', 'Religion', 'Death',
'Assent', 'Nonfluencies', 'Fillers', 'Total first person', 'Total third person', 'Positive feelings',
'Optimism and energy', 'Communication', 'Other references to people', 'Up', 'Down', 'Occupation', 'School',
'Sports', 'TV','Music','Metaphysical issues', 'Physical states and functions', 'Sleeping', 'Grooming']

def get_liwc_features(train_data, test_data):
    """
        Creates a LIWC feature extractor.
        NOTE: this function is currently not being used in this program.
    """
    print("getting liwc features")
    train_liwc_matrix = []
    test_liwc_matrix = []
    for phrase in train_data:
        liwc_scores = word_category_counter.score_text(phrase)
        feature_vector = []
        for key in liwc_categories:
            if key in liwc_scores.keys():
                # print(key)
                # print(liwc_scores[key])
                feature_vector.append(liwc_scores[key])
            else:
                feature_vector.append(0)
        # print(feature_vector)
        train_liwc_matrix.append(feature_vector)
    for phrase in test_data:
        liwc_scores = word_category_counter.score_text(phrase)
        feature_vector = []
        for key in liwc_categories:
            if key in liwc_scores.keys():
                # print(key)
                # print(liwc_scores[key])
                feature_vector.append(liwc_scores[key])
            else:
                feature_vector.append(0)
        test_liwc_matrix.append(feature_vector)
    # print(train_liwc_matrix)
    return sparse.csr_matrix(train_liwc_matrix), sparse.csr_matrix(test_liwc_matrix)
  

def get_ngram_features(train_data, test_data):
    """
        Creates a bag of words unigram/bigram feature extractor.
        Fits the extractor to the training data, then applies the extractor to both
            the training data and the test data.
        Parameters are the training instances and testing instances (just the text phrases)
            as Series.
        Returns the extracted feature sets of the training and test data, as matrices.
    """
    print("getting ngram features")
    ngram_vectorizer = CountVectorizer(ngram_range = (1, 2))
    ngram_vectorizer = ngram_vectorizer.fit(train_data)
    return ngram_vectorizer.transform(train_data), ngram_vectorizer.transform(test_data)


def get_idf_features(train_data, test_data):
    """
        Creates a tfidf unigram/bigram feature extractor.
        Fits the extractor to the training data, then applies the extractor to both
            the training data and the test data.
        Parameters are the training instances and testing instances (just the text phrases)
            as Series.
        Returns the extracted feature sets of the training and test data, as matrices.
    """
    tfidf = TfidfVectorizer(tokenizer = tokenize, ngram_range = (1, 2))
    tfidf.fit(train_data)
    return tfidf.transform(train_data), tfidf.transform(test_data)


def get_all_features(train_data, test_data):
    """
        Calls all feature extractor methods to obtain the different feature sets.
        Parameters are the training instances and testing instances (just the text phrases)
            as Series.
        Returns the combined extracted feature sets of the training and test data, as a matrix.
    """
    #train_wc_matrix, test_wc_matrix = get_word_count_features(train_data, test_data)
    train_idf_matrix, test_idf_matrix = get_idf_features(train_data, test_data)
    train_ngram_matrix, test_ngram_matrix = get_ngram_features(train_data, test_data)
    # train_liwc_matrix, test_liwc_matrix = get_liwc_features(train_data, test_data)
    return sparse.hstack([train_idf_matrix, train_ngram_matrix]), \
        sparse.hstack([test_idf_matrix, test_ngram_matrix])
    

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
    except FileNotFoundError:
        print("Error: Training file does not exist. File must be of type csv")
        sys.exit(1)
    except:
        print("Error: Unknown error occurred trying to read train data file")
        sys.exit(1)
    try:
        test_data_df = pandas.read_csv(args.testFile)
    except FileNotFoundError:
        print("Error: Testing file does not exist. File must be of type csv")
        sys.exit(1)
    except:
        print("Error: Unknown error occurred trying to read test data file")
        sys.exit(1)

        
    #### preprocessing & feature extraction ####
    train_feature_set, test_feature_set = get_all_features(train_data_df["Phrase"], test_data_df["Phrase"])
    print("finished getting features")

    # training    
    model = trainClassifiers(train_feature_set, train_data_df["Sentiment"].tolist())
    print("finished training")
    # test
    predictions_df = pandas.DataFrame(model.predict(test_feature_set))
    predictions_df = pandas.concat([test_data_df["PhraseId"], predictions_df], axis = 1)
    predictions_df.to_csv(path_or_buf = args.outFile, header = ["PhraseId", "Sentiment"], index = False)
    
    # write performance stats to txt file
    #perf_out_file = open(args.perfFile, "w")
    #performance_metrics.get_performance_train(model, train_feature_set, train_data_df["Sentiment"].tolist(), perf_out_file, True)
    #performance_metrics.get_performance_cv(model, train_feature_set, train_data_df["Sentiment"].tolist(), perf_out_file, 3)
    #perf_out_file.close()

    
if __name__ == '__main__':
    main()
