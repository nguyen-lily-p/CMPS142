import sklearn.metrics
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def get_performance_train(model, features, labels, file, class_performance = False):
    """
        Get the performance metrics of a model on the direct, entire training set.
        Parameters are the ensemble model, the extracted feature set of the training data,
            the labels of the instances, the file to write results to, and whether
            to print the statistics for the individual classes/labels.
    """
    print("in get_performance")
    file.write("\n******************** TRAINING SET ********************\n")
    
    # performance of individual classifiers
    classifierList = model.estimators_
    for classifier in classifierList:
        get_performance_train_helper(classifier, features, labels, file, class_performance)
    
    # performance of the ensemble classifier
    get_performance_train_helper(model, features, labels, file, class_performance)
    
    # count of the instances in each class
    predictions = model.predict(features)
    class_metrics = sklearn.metrics.precision_recall_fscore_support(labels, predictions)
    file.write("\n\tClass Count:\n")
    for i in range(0, 5):
        file.write("\t\tClass " + str(i) + ": ")
        file.write(str(class_metrics[3][i]) + "\n")
        
    file.write("\n************************************************************\n")

    
def get_performance_train_helper(classifier, features, labels, file, class_performance):
    """
        Helper function for get_performance_train()
        Prints the accuracy, precision, recall, F1-score, and confusion matrix of a classifier.
        If class_performance = True, also prints the precision, recall, and F1-score
            for each class/label.
    """
    name = str(type(classifier))[16:-2]
    file.write("\n********** CLASSIFIER: " + name + "**********\n")
    
    predictions = classifier.predict(features)
    class_metrics = sklearn.metrics.precision_recall_fscore_support(labels, predictions)
    
    # accuracy
    file.write("\tAccuracy: " + str(sklearn.metrics.accuracy_score(labels, predictions)) + "\n")
    
    # precision
    file.write("\tPrecision: " + str(sklearn.metrics.precision_score(labels, predictions, average = "weighted")) + "\n")
    if class_performance:
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[0][i]) + "\n")

    # recall
    file.write("\tRecall: " + str(sklearn.metrics.recall_score(labels, predictions, average = "weighted")) + "\n")
    if class_performance:
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[1][i]) + "\n")

    # f1-score
    file.write("\tF-1 Score: " + str(sklearn.metrics.f1_score(labels, predictions, average = "weighted")) + "\n")
    if class_performance:
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[2][i]) + "\n")
   
    # confusion matrix
    file.write("\tConfusion Matrix:\n")
    conf_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    for row in conf_matrix:
        file.write("\t\t" + str(row) + "\n")
        

def get_performance_cv(model, features, labels, file, cv_val = 5):
    """
        Get the performance metrics of a model using cross validation.
        Parameters are the ensemble model, the extracted feature set of the training data,
            the labels of the instances, the file to write results to, and k-fold number.
        NOTE: accuracy, precision, and recall metrics are currently commented out because
            program takes too long to run when computing all of them.
    """
    file.write("\n******************** CROSS VALIDATION (CV = " + str(cv_val) + ") ********************\n")
    
    classifierList = model.estimators_
    classifierList.append(model) # add VotingClassifier to list
    
    # output performance of individual classifiers and ensemble classifier
    for classifier in classifierList:
        name = str(type(classifier))[16:-2]
        file.write("\n********** CLASSIFIER: " + name + "**********\n")
        #file.write("\tAccuracy:  " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "accuracy")) + "\n")
        #file.write("\tPrecision: " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "precision_weighted")) + "\n")
        #file.write("\tRecall:    " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "recall_weighted")) + "\n")
        file.write("\tF-1 Score: " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "f1_weighted")) + "\n") 

    file.write("\n************************************************************\n")
