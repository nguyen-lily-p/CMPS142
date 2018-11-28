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


def main():
    # read in command-line arguments, if any
    parser = argparse.ArgumentParser(description="program description")
    parser.add_argument("--test", dest = "testFile", default = TESTING_DATA_PATH, \
            type = str, help = "the path to the .csv file containing the test data")
    parser.add_argument("--out", dest = "outFile", default = OUTPUT_PATH, \
            type = str, help = "the path of the output file")
    args = parser.parse_args()
    print(args.testFile)

    # read in training data

    # preprocessing

    # feature extraction

    # training - send to different algorithms

    # test

if __name__ == '__main__':
    main()
