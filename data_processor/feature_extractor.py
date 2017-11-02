import os
from PIL import Image
import cv2
import logging

from utils.utils import DATA_SET_TRAIN_POSITIVE_, DATA_SET_TRAIN_NEGATIVE_, FEATURES_LOCATION, save_features_in_csv, \
    get_largest_contour, get_contour_arch_length, get_contour_area, get_eccentricity, get_contour_moments, \
    INF_COMPACTNESS_VALUE


def get_features_from_image(image):
    """
    Return features of the largest contour (Assuming it will be the bacteria in this image or background.
    :param image: Input image.
    :return: list of features.
    """
    largest_contour = get_largest_contour(image)
    if largest_contour is not None:
        contour_area = get_contour_area(largest_contour)
        contour_arch_length = get_contour_arch_length(largest_contour)
        if contour_area > 0:
            compactness = contour_arch_length / contour_area
        else:
            return None
        eccentricity = get_eccentricity(largest_contour)
        moments = get_contour_moments(largest_contour)
        features = {"moments": moments, "compactness": compactness, "eccentricity": eccentricity}
        return features
    else:
        return None


def extract_features():
    label = 0  # negative image.
    ans = 0
    for file_name in os.listdir(DATA_SET_TRAIN_NEGATIVE_):
        if file_name.endswith(".png"):
            image = cv2.imread(DATA_SET_TRAIN_NEGATIVE_ + file_name)
            try:
                features_list = get_features_from_image(image)
                if features_list is None:
                    continue
                save_features_in_csv(features_list, label, FEATURES_LOCATION)
                ans += 1
            except:
                logging.debug("Error getting features for " + file_name)
    label = 1  # positive images.
    for file_name in os.listdir(DATA_SET_TRAIN_POSITIVE_):
        if file_name.endswith(".png"):
            image = cv2.imread(DATA_SET_TRAIN_POSITIVE_ + file_name)
            try:
                features_list = get_features_from_image(image)
                if features_list is None:
                    continue
                save_features_in_csv(features_list, label, FEATURES_LOCATION)
                ans += 1
            except:
                logging.debug("Error getting features for " + file_name)
    print ans

if __name__ == '__main__':
    extract_features()
