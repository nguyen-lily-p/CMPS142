import sys, getopt, pandas


# file path for training data
TRAINING_DATA_PATH = "knn_train.csv"
# file path for test data
TESTING_DATA_PATH = "knn_test.csv"


def get_options():
    """ 
        Analyze command line options and arguments
        Sets the values of k_val and l_val based on the arguments
        Or prints an error message and exits if options or arguments are incorrect
    """
    # read in command line options/arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["K=", "k=", "Method=", "method="])
    except getopt.GetoptError as opt_error:
        print("Error:", str(opt_error))
        sys.exit(2)
    
    k_flag = False
    method_flag = False

    # check for "--K" and "--method" options and for valid arguments
    for opt in opts:
        if (opt[0]).upper() == "--K":
            try:
                k_val = int(opt[1])
            except ValueError:
                print("Error: Argument for '--K' option must be an integer")
                sys.exit(2)
            if k_val < 1:
                print("Error: Argument value for '--K' must be positive")
                sys.exit(2)
            k_flag = True
        elif (opt[0]).lower() == "--method":
            if (opt[1]).upper() == "L1":
                l_val = 1
            elif (opt[1]).upper() == "L2":
                l_val = 2
            elif (opt[1]).upper() == "LINF":
                l_val = 0
            else:
                print("Error: Argument value for '--method' must be either L1, L2, or Linf")
                sys.exit(2)
            method_flag = True

    if not k_flag:
        print("Error: Missing '--K' option")
    if not method_flag:
        print("Error: Missing '--method' option")
    if not k_flag or not method_flag:
        sys.exit(2)

    return k_val, l_val


def read_data(filepath):
    """ 
        Reads in the training data from the given csv file 
        and returns the data as a pandas dataframe
    """
    try:
        df = pandas.read_csv(filepath)
    except FileNotFoundError as err:
        print("Error:", str(err))
        sys.exit(1)
    except pandas.io.common.CParserError as err:
        print("Error:", str(err), "File must be of type csv")
        sys.exit(1)
    except:
        print("Error: Unknown error occurred trying to read data file")
        sys.exit(1)
    
    # make indexing start at 1
    df.index = df.index + 1

    return df


def get_neighbors_L2(test_instance, training_df, k_val):
    """
        Finds the k nearest neighbors in the training instances of the given test instance
            using the L2 distance algorithm
    """
    neighbors = [(-1, float("inf"), 0)] * k_val

    for row in training_df.itertuples():
        # calculate distance
        dist = 0
        for i in range(1, len(training_df.columns)):
            dist += (test_instance[i] - row[i]) ** 2
        
        # update list of neighbors, if necessary
        if dist <  (neighbors[len(neighbors) - 1])[1]:
            neighbors.pop()
            neighbors.append((row[0], dist, row[len(training_df.columns)]))
            neighbors.sort(key=lambda x: x[1])
    
    return neighbors


def main():

    k_val, l_val = get_options()
    print ("K Val:", k_val)
    print ("L Val:", l_val)

    training_df = read_data(TRAINING_DATA_PATH)
    test_df = read_data(TESTING_DATA_PATH)
    print (test_df)
    print (len(test_df.columns))

    # run through test data
    print("\nTEST INSTANCE |")
    #print("\n{0:13s} | {1:80s} | {2:10s} | {3:6s}".format("TEST INSTANCE", "K NEAREST NEIGHBORS", "PREDICTION", "ACTUAL"))
    for instance in test_df.itertuples(True, None):
        neighbors = get_neighbors_L2(instance, training_df, k_val)
        print("{0:13d} | {1:36s} | {2:s}".format(instance[0], "KNN (Instance Num, Dist, Label)", str(neighbors)))
        print("{0:13s} | {1:36s} | {2:s}".format(" ", "Prediction", "?"))
        print("{0:13s} | {1:36s} | {2:d}".format(" ", "Actual", instance[len(test_df.columns)]))
        #print("{0:13d} | {1:80s} | {2:10s} | {3:6d}".format(instance[0], str(neighbors), "?", instance[len(test_df.columns)]))


    #call appropriate function based on l_val
    #read in test data

main()
