from sklearn.neural_network import MLPClassifier
from utils.utils import get_features_labels_from_csv


def mlp_classifier(features):
    train_features, labels = get_features_labels_from_csv()
    clf = MLPClassifier(hidden_layer_sizes=(22, 11, 5), random_state=1)
    clf.fit(train_features, labels)
    return clf.predict([features])
