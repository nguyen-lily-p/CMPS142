import sys, getopt, pandas

# TODO: create function to compare first test instance vs first training instance


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
        opts, args = getopt.getopt(sys.argv[1:], "", ["K=", "k=", "Method=", \
            "method="])
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
                print("Error: Argument value for '--method' must be either L1, \
                        L2, or Linf")
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


def get_neighbors(test_instance, training_df, k_val, l_val):
    """
        Finds the k nearest neighbors in the training instances of the given test instance
            using the L2 distance algorithm
    """
    neighbors = [(-1, float("inf"), 0)] * k_val

    for row in training_df.itertuples():
        dist = 0

        # calculate distance
        if l_val == 2:
            dist = get_distance_L2(test_instance, row, len(training_df.columns))
        elif l_val == 1:
            dist = get_distance_L1(test_instance, row, len(training_df.columns))
        elif l_val == 0:
            dist = get_distance_Linf(test_instance, row, len(training_df.columns))
        
        # update list of neighbors, if necessary
        if dist <  (neighbors[len(neighbors) - 1])[1]:
            neighbors.pop()
            neighbors.append((row[0], dist, row[len(training_df.columns)]))
            neighbors.sort(key=lambda x: x[1])
    
    return neighbors


def get_distance_L2(test_instance, training_instance, length):
    """
        Calculates and returns the L2 distance between the test_instance and 
            the training_instance
        length is the number of dimensions the points
    """
    dist = 0
    for i in range(1, length):
        dist += (test_instance[i] - training_instance[i]) ** 2

    return dist


def get_distance_L1(test_instance, training_instance, length):
    """
        Calculates and returns the L1 distance between the test_instance and 
            training_instance
        length is the number of dimensions of the points
    """
    dist = 0
    for i in range(1, length):
        dist += abs(test_instance[i] - training_instance[i])

    return dist


def get_distance_Linf(test_instance, training_instance, length):
    """
        Calculates and returns the L-inf (max norm) distance between the test_instance 
            and training_instance
        length is the number of dimensions of the points
    """
    dist_list = []
    for i in range(1, length):
        dist_list.append(abs(test_instance[i] - training_instance[i]))

    return max(dist_list)


def get_prediction(neighbors):
    """
        Predicts a label (-1 or 1) based on the labels of all the given neighbors
        NOTE: May not work correctly for an even k/number of neighbors
    """
    pred = 0
    for n in neighbors:
        pred += n[2]

    if pred < 0:
        return -1
    elif pred > 0:
        return 1
    else:
        return 0


def get_stats(tp, tn, fp, fn):
    """
        Calculates and prints the performance statisics:
        confusion matrix, accuracy, precision, recall, and f-measure
    """

    print ("\nCONFUSION MATRIX")
    print ("TP:{0:4d}| FP:{1:4d}".format(tp, fp))
    print ("FN:{0:4d}| FP:{1:4d}\n".format(fn, tn))

    print ("{0:9s}: {1:5d}/{2:5d} = {3:f}%".format("ACCURACY", tp + tn, 
        tp + tn + fp + fn, (tp + tn)/(tp + tn + fp + fn) * 100))

    precision = tp / (tp + fp)
    print ("{0:9s}: {1:5d}/{2:5d} = {3:f}%".format("PRECISION", tp, tp + fp, 
        precision * 100))

    recall = tp / (tp + fn)
    print ("{0:9s}: {1:5d}/{2:5d} = {3:f}%".format("RECALL", tp, tp + fn, 
        recall * 100))

    print ("{0:9s}: {1:.3f}/{2:.3f} = {3:f}%".format("F-MEASURE", 
        2 * precision * recall, precision + recall, 
        (2 * precision * recall)/(precision + recall) * 100))


def main():

    k_val, l_val = get_options()

    training_df = read_data(TRAINING_DATA_PATH)
    test_df = read_data(TESTING_DATA_PATH)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    print("\nTEST INSTANCE |")
    
    # run through test data
    for instance in test_df.itertuples(True, None):
        neighbors = get_neighbors(instance, training_df, k_val, l_val)
        prediction = get_prediction(neighbors)

        if prediction > 0:
            if prediction == instance[len(test_df.columns)]:
                true_pos += 1
            else:
                false_pos += 1
        elif prediction < 0:
            if prediction == instance[len(test_df.columns)]:
                true_neg += 1
            else:
                false_neg += 1


        print("{0:13d} | {1:36s} | {2:s}".format(instance[0], 
            "KNN (Instance Num, Dist, Label)", str(neighbors)))
        print("{0:13s} | {1:36s} | {2:d}".format(" ", "Prediction", prediction))
        print("{0:13s} | {1:36s} | {2:d}".format(" ", "Actual", 
            instance[len(test_df.columns)]))

    # print stats
    get_stats(true_pos, true_neg, false_pos, false_neg)

main()
