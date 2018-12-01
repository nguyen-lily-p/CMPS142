import pandas, main
from sklearn.feature_extraction.text import TfidfVectorizer

def compare_models(train_data_df):
    for i in range (1, 6):
        tfidf = TfidfVectorizer(tokenizer = main.tokenize, min_df = (i / 1000000.0))
        feature_set = tfidf.fit_transform(train_data_df["Phrase"])
        model = main.trainClassifiers(feature_set, train_data_df["Sentiment"])
        print("\nMODEL: MIN_DF = ", (i / 1000000.0))
        print(model.score(feature_set, train_data_df["Sentiment"]))
        print("(Instances x Features): ", feature_set.shape)
