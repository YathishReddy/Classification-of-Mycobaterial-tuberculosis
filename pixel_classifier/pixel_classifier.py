import os
from PIL import Image
import numpy as np

from utils.utils import DATA_SET_TRAIN_POSITIVE, DATA_SET_TRAIN_NEGATIVE, PIXEL_CLASSIFIER_TEST_INPUT, TEST_FILE_1
from classifiers.naive_bayes import naive_bayes_classifier


def get_flat_image(image):
    """
    Returns a 1D image.
    :param image: A 2D image.
    :return: A 1D image.
    """
    data = np.asarray(image, dtype="int32")
    height, width = data.shape[0], data.shape[1]
    # 4 channels.
    flat = data.reshape(height * width, 4)
    return flat.tolist()


def get_positive_images():
    """
    Get flatten (1D) images from negative data set.
    :return:
    Array containing negative images as list.
    """
    positive_images = []
    for file_name in os.listdir(DATA_SET_TRAIN_POSITIVE):
        if file_name.endswith(".png"):
            image = Image.open(DATA_SET_TRAIN_POSITIVE + file_name)
            flat_image = get_flat_image(image)
            positive_images.extend(flat_image)
    return positive_images


def get_negative_images():
    """
    Get flatten (1D) images from negative data set.
    :return:
    Array containing negative images as list.
    """
    negative_images = []
    for filename in os.listdir(DATA_SET_TRAIN_NEGATIVE):
        if filename.endswith(".png"):
            image = Image.open(DATA_SET_TRAIN_NEGATIVE + filename)
            flat_image = get_flat_image(image)
            negative_images.extend(flat_image)
    return negative_images


def get_segmented_image(file_loc, classifier):
    image = Image.open(file_loc)
    image = np.asarray(image, dtype="int32")
    segmented_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for h in xrange(image.shape[0]):
        for w in xrange(image.shape[1]):
            list_ = [image[h][w][0], image[h][w][1], image[h][w][2], 255]
            data_ = np.array(list_)
            data_ = data_.reshape(1, -1)
            predicted_val = classifier.predict(data_)
            segmented_image[h][w] = predicted_val * 255
    return Image.fromarray(segmented_image)


def pixel_classifier():
    """
    Take both positive and negative images. Flatten them and train against naive bayes classifier and linear
    classifiers.
    :return:
    """
    positive_images = get_positive_images()
    negative_images = get_negative_images()
    classifier = naive_bayes_classifier(positive_images=positive_images, negative_images=negative_images)
    segmented_image = get_segmented_image(PIXEL_CLASSIFIER_TEST_INPUT + TEST_FILE_1, classifier)
    segmented_image.show()

if __name__ == '__main__':
    pixel_classifier()
