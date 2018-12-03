import sklearn.metrics
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def get_performance(model, features, labels, output_file):
    """

    """
    print("in get_performance")
    file = open(output_file, "w")
		
    ### output performance of individual classifiers ###
    classifierList = model.estimators_
    for classifier in classifierList:
        name = str(type(classifier))[16:-2]
        file.write("\n********** CLASSIFIER: " + name + "**********\n")
        
        # performance metrics with cross validation
        file.write("\nCROSS VALIDATION (CV = 5)\n")
        file.write("\tAccuracy:  " + str(cross_val_score(classifier, features, labels, cv = 5, scoring = "accuracy")) + "\n")
        file.write("\tPrecision: " + str(cross_val_score(classifier, features, labels, cv = 5, scoring = "precision_weighted")) + "\n")
        file.write("\tRecall:    " + str(cross_val_score(classifier, features, labels, cv = 5, scoring = "recall_weighted")) + "\n")
        file.write("\tF-1 Score: " + str(cross_val_score(classifier, features, labels, cv = 5, scoring = "f1_weighted")) + "\n")

        # performance metrics on training set
        file.write("\nTRAINING SET\n")
        predictions = classifier.predict(features)
        class_metrics = sklearn.metrics.precision_recall_fscore_support(labels, predictions)
        
        file.write("\tAccuracy: " + str(sklearn.metrics.accuracy_score(labels, predictions)) + "\n")
        
        file.write("\tPrecision:\n")		
        file.write("\t\tAverage: " + str(sklearn.metrics.precision_score(labels, predictions, average = "weighted")) + "\n")
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[0][i]) + "\n")
        
        file.write("\tRecall:\n")
        file.write("\t\tAverage: " + str(sklearn.metrics.recall_score(labels, predictions, average = "weighted")) + "\n")
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[1][i]) + "\n")
        
        file.write("\tF-1 Score:\n")
        file.write("\t\tAverage: " + str(sklearn.metrics.f1_score(labels, predictions, average = "weighted")) + "\n")
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[2][i]) + "\n")
        
        file.write("\tConfusion Matrix:\n")
        conf_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
        for row in conf_matrix:
            file.write("\t\t" + str(row) + "\n")

        file.write("\tClass Count:\n")
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[3][i]) + "\n")
        

    ### output performance of VotingClassifier ###
    file.write("\n********** CLASSIFIER: VOTING ENSEMBLE **********\n")
    
    # performance metrics with cross validation
    file.write("\nCROSS VALIDATION (CV = 5)\n")
    file.write("\tAccuracy: " + str(cross_val_score(model, features, labels, cv = 5, scoring = "accuracy")) + "\n")
    file.write("\tPrecision: " + str(cross_val_score(model, features, labels, cv = 5, scoring = "precision_weighted")) + "\n")
    file.write("\tRecall:    " + str(cross_val_score(model, features, labels, cv = 5, scoring = "recall_weighted")) + "\n")
    file.write("\tF-1 Score: " + str(cross_val_score(model, features, labels, cv = 5, scoring = "f1_weighted")) + "\n")

    # performance metrics on training set
    file.write("\nTRAINING SET\n")
    predictions = classifier.predict(features)
    class_metrics = sklearn.metrics.precision_recall_fscore_support(labels, predictions)
        
    file.write("\tAccuracy: " + str(sklearn.metrics.accuracy_score(labels, predictions)) + "\n")
        
    file.write("\tPrecision:\n")		
    file.write("\t\tAverage: " + str(sklearn.metrics.precision_score(labels, predictions, average = "weighted")) + "\n")
    for i in range(0, 5):
        file.write("\t\tClass " + str(i) + ": ")
        file.write(str(class_metrics[0][i]) + "\n")
        
    file.write("\tRecall:\n")
    file.write("\t\tAverage: " + str(sklearn.metrics.recall_score(labels, predictions, average = "weighted")) + "\n")
    for i in range(0, 5):
        file.write("\t\tClass " + str(i) + ": ")
        file.write(str(class_metrics[1][i]) + "\n")

    file.write("\tF-1 Score:\n")
    file.write("\t\tAverage: " + str(sklearn.metrics.f1_score(labels, predictions, average = "weighted")) + "\n")
    for i in range(0, 5):
        file.write("\t\tClass " + str(i) + ": ")
        file.write(str(class_metrics[2][i]) + "\n")

    file.write("\tConfusion Matrix:\n")
    conf_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    for row in conf_matrix:
        file.write("\t\t" + str(row) + "\n")

    file.write("\tClass Count:\n")
    for i in range(0, 5):
        file.write("\t\tClass " + str(i) + ": ")
        file.write(str(class_metrics[3][i]) + "\n")

    file.close()
