import sys, getopt

def get_training_data():




def main():
    # read in command line options/arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["K=", "k=", "Method=", "method="])
    except getopt.GetoptError as opt_error:
        print ("Error:", str(opt_error))
        sys.exit(2)
    
    # check for "--K" and "--method" options
    k_flag = False
    method_flag = False

    for opt in opts:
        if (opt[0]).upper() == "--K":
            try:
                k_val = int(opt[1])
            except ValueError:
                print ("Error: Argument for '--K' option must be an integer")
                sys.exit(2)
            if k_val < 1:
                print ("Error: Argument value for '--K' must be positive")
                sys.exit(2)
            k_flag = True
        elif (opt[0]).lower() == "--method":
            if (opt[1]).upper() == "L1":
                # set value/flag
            elif (opt[1]).upper() == "L2":
                # set value/flag
            elif (opt[1]).upper() == "LINF":
                # set value/flag
            else:
                print ("Error: Argument value for '--method' must be either L1, L2, or Linf")
                sys.exit(2)
            method_flag = True

    if not k_flag:
        print ("Error: Missing '--K' option")
    if not method_flag:
        print ("Error: Missing '--method' option")
    if not k_flag or not method_flag:
        sys.exit(2)

    # validate option arguments
    # check that arg for --K is an integer
    # check that arg for --method is one of the allowable options
    print (opts)
    print (args)



main()
