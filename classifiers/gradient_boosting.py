from sklearn.ensemble import GradientBoostingClassifier
from utils.utils import get_features_labels_from_csv


def gradient_boosting_classifier(features):
    train_features, labels = get_features_labels_from_csv()
    bdt = GradientBoostingClassifier(n_estimators=250)
    bdt.fit(train_features, labels)
    return bdt.predict([features])
