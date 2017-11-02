from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.utils import get_features_labels_from_csv


def adaboost_classifier(features):
    train_features, labels = get_features_labels_from_csv()
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
    bdt.fit(train_features, labels)
    return bdt.predict([features])
