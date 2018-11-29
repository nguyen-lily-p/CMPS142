# Brianna Atayan -  - batayan@ucsc.edu
# Colin Maher    - - csmaher@ucsc.edu
# Lily Nguyen    - 1596857 - lnguye78@ucsc.edu

import argparse

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
    

def main():
    # read in training data
    parser = argparse.ArgumentParser(description="program description")
    parser.add_argument("--test", dest = "testFile", default = TESTING_DATA_PATH, \
            type = str, help = "the path to the .csv file containing the test data")
    parser.add_argument("--out", dest = "outFile", default = OUTPUT_PATH, \
            type = str, help = "the path of the output file")
    args = parser.parse_args()
    print(args.testFile)

    # preprocessing

    # feature extraction

    # training - send to different algorithms

    # test

if __name__ == '__main__':
    main()
