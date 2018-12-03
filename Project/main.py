# Brianna Atayan - 1632743 - batayan@ucsc.edu
# Colin Maher    - 1432169 - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse, csv, pandas, sys, string, numpy, compare_models, sklearn.metrics, performance_metrics, word_category_counter
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


# declares classifies that will be trained and used for testing
# global so all functions can access
naiveBayesModel = MultinomialNB()
linearSVCModel = LinearSVC()
logRegModel = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', random_state = 1, max_iter=1000)

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

def get_pos_features(data):
    pass


def get_unigram_bow_features(train_data, test_data):
    vectorizer = CountVectorizer() 
    vectorizer = vectorizer.fit(train_data)
    return vectorizer.transform(train_data), vectorizer.transform(test_data)


def get_idf_features(train_data, test_data):
    tfidf = TfidfVectorizer(tokenizer = tokenize)
    tfidf.fit(train_data)
    return tfidf.transform(train_data), tfidf.transform(test_data)

def get_all_features(train_data, test_data):
    #train_wc_matrix, test_wc_matrix = get_word_count_features(train_data, test_data)
    train_uni_bow_matrix, test_uni_bow_matrix = get_unigram_bow_features(train_data, test_data)
    train_idf_matrix, test_idf_matrix = get_idf_features(train_data, test_data)
    train_liwc_matrix, test_liwc_matrix = get_liwc_features(train_data, test_data)
    #return sparse.hstack([train_idf_matrix, train_wc_matrix, train_uni_bow_matrix]), \
    #       sparse.hstack([test_idf_matrix, test_wc_matrix, test_uni_bow_matrix])
    return sparse.hstack([train_idf_matrix, train_uni_bow_matrix, train_liwc_matrix]), \
           sparse.hstack([test_idf_matrix, test_uni_bow_matrix, test_liwc_matrix])
    
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

    # training    
    model = trainClassifiers(train_feature_set, train_data_df["Sentiment"].tolist())
    print("finished training")
    # test
    predictions_df = pandas.DataFrame(model.predict(test_feature_set[:20000]))
    predictions_df = pandas.concat([test_data_df["PhraseId"], predictions_df], axis = 1)
    predictions_df.to_csv(path_or_buf = args.outFile, header = ["PhraseId", "Sentiment"], index = False)
    performance_metrics.get_performance(model, test_feature_set, test_data_df["Sentiment"].tolist(), args.perfFile)
    

if __name__ == '__main__':
    main()
