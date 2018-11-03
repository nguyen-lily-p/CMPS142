import sys, getopt, pandas

# file path for training data
TRAINING_DATA_PATH = "knn_train.csv"

def get_options(k_val, l_val):
    """ Analyze command line options and arguments
        Sets the values of k_val and l_val based on the arguments
            or prints an error message and exits
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

def read_training_data(filepath):
    try:
        training_df = pandas.read_csv(filepath)
    except FileNotFoundError as err:
        print("Error:", str(err))
        sys.exit(2)

    print(training_df)


def main():
    k_val = None
    l_val = None
    get_options(k_val, l_val)
    read_training_data(TRAINING_DATA_PATH)

    #read in training data
    #call appropriate function based on l_val
    #read in test data

main()
