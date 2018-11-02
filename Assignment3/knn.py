import sys, getopt

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "", ["K=", "k=", "Method=", "method="])
    except getopt.GetoptError as opt_error:
        print ("Error:", str(opt_error))
        sys.exit(2)
    
    # check for "--K" and "--method" options
    k_flag = False
    method_flag = False
    for opt in opts:
        if (opt[0]).lower() == "--k":
            # set value of K
            k_flag = True
        elif (opt[0]).lower() == "--method":
            # choose method function
            method_flag = True

    if not k_flag:
        print ("Error: Missing '--K' option")
    if not method_flag:
        print ("Error: Missing '--method' option")
    if not k_flag or not method_flag:
        sys.exit(2)

    # validate option arguments
    print (opts)
    print (args)

main()
