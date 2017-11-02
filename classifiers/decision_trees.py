from sklearn import tree

from utils.utils import get_features_labels_from_csv


def decision_tree_classifier(features):
    train_features, labels = get_features_labels_from_csv()
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(train_features, labels)
    return classifier.predict([features])
