from utils.utils import get_features_labels_from_csv
import sklearn.naive_bayes as nb


def classify_region_of_interest(features):
    train_features, labels = get_features_labels_from_csv()
    naive_bayes = nb.GaussianNB()
    classifier = naive_bayes.fit(train_features, labels)
    return classifier.predict([features])
