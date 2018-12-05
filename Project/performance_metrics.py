import sklearn.metrics
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def get_performance_train(model, features, labels, file, class_performance = False):
    file.write("\n******************** TRAINING SET ********************\n")
    
    # performance of individual classifiers
    classifierList = model.estimators_
    for classifier in classifierList:
        get_performance_train_helper(classifier, features, labels, file, class_performance)
    
    get_performance_train_helper(model, features, labels, file, class_performance)
    
    predictions = model.predict(features)
    class_metrics = sklearn.metrics.precision_recall_fscore_support(labels, predictions)
    file.write("\n\tClass Count:\n")
    for i in range(0, 5):
        file.write("\t\tClass " + str(i) + ": ")
        file.write(str(class_metrics[3][i]) + "\n")
        
    file.write("\n************************************************************\n")

def get_performance_train_helper(classifier, features, labels, file, class_performance):
    name = str(type(classifier))[16:-2]
    file.write("\n********** CLASSIFIER: " + name + "**********\n")
    
    predictions = classifier.predict(features)
    class_metrics = sklearn.metrics.precision_recall_fscore_support(labels, predictions)
    
    file.write("\tAccuracy: " + str(sklearn.metrics.accuracy_score(labels, predictions)) + "\n")
    
    file.write("\tPrecision: " + str(sklearn.metrics.precision_score(labels, predictions, average = "weighted")) + "\n")
    if class_performance:
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[0][i]) + "\n")

    file.write("\tRecall: " + str(sklearn.metrics.recall_score(labels, predictions, average = "weighted")) + "\n")
    if class_performance:
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[1][i]) + "\n")

    file.write("\tF-1 Score: " + str(sklearn.metrics.f1_score(labels, predictions, average = "weighted")) + "\n")
    if class_performance:
        for i in range(0, 5):
            file.write("\t\tClass " + str(i) + ": ")
            file.write(str(class_metrics[2][i]) + "\n")
   
    file.write("\tConfusion Matrix:\n")
    conf_matrix = sklearn.metrics.confusion_matrix(labels, predictions)
    for row in conf_matrix:
        file.write("\t\t" + str(row) + "\n")
        

def get_performance_cv(model, features, labels, file, cv_val = 5):
    
    file.write("\n******************** CROSS VALIDATION (CV = " + str(cv_val) + ") ********************\n")
    
    classifierList = model.estimators_
    classifierList.append(model) # add VotingClassifier to list
    
    ### output performance of individual classifiers and ensemble classifier ###
    for classifier in classifierList:
        name = str(type(classifier))[16:-2]
        file.write("\n********** CLASSIFIER: " + name + "**********\n")
        #file.write("\tAccuracy:  " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "accuracy")) + "\n")
        #file.write("\tPrecision: " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "precision_weighted")) + "\n")
        #file.write("\tRecall:    " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "recall_weighted")) + "\n")
        file.write("\tF-1 Score: " + str(cross_val_score(classifier, features, labels, cv = cv_val, scoring = "f1_weighted")) + "\n") 

    file.write("\n************************************************************\n")
