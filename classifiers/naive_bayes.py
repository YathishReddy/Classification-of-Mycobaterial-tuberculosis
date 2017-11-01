import sklearn.naive_bayes as nb


def naive_bayes_classifier(positive_images, negative_images):
    """
    A Naive bayes classifier trained with features as 4 channels and labels as 1 for positive image and 0 for
        negative image.
    :param positive_images: list containing positive images.
    :param negative_images: list containing negative images.
    :return: A naive bayes classifier.
    """
    naive_bayes = nb.GaussianNB()
    # Prepare input/features for naive bayes.
    all_images = []
    all_images.extend(positive_images)
    all_images.extend(negative_images)
    labels = [1]*len(positive_images) + [0]*len(negative_images)
    classifier_naive_bayes = naive_bayes.fit(all_images, labels)
    return classifier_naive_bayes
