import cv2
import csv
import logging

DATA_SET_DIR = "/home/mohit/Classification-of-Mycobaterial-tuberculosis/datasets/"

DATA_SET_TRAIN = DATA_SET_DIR + "train/"

DATA_SET_TRAIN_POSITIVE = DATA_SET_TRAIN + "batch_positive/"
DATA_SET_TRAIN_NEGATIVE = DATA_SET_TRAIN + "batch_negative/"

DATA_SET_TRAIN_POSITIVE_ = DATA_SET_TRAIN + "positive/"
DATA_SET_TRAIN_NEGATIVE_ = DATA_SET_TRAIN + "negative/"

PIXEL_CLASSIFIER_TEST_INPUT = DATA_SET_DIR + "test/"

TEST_FILE_1 = "test1.png"
LARGE_TEST_FILE = "test2.png"

INF_COMPACTNESS_VALUE = 10.0

FEATURES_LOCATION = "/home/mohit/Classification-of-Mycobaterial-tuberculosis/features.csv"
FEATURES_ORDER = ["compactness", "eccentricity", "nu11", "nu12", "nu21", "nu02", "nu20", "nu03", "nu30"]


def save_features_in_csv(features_list, label, csv_file_loc):
    features = []
    for feature in FEATURES_ORDER:
        if feature in features_list:
            features.append(features_list[feature])
        else:
            features.append(features_list["moments"][feature])
    features.append(label)
    with open(csv_file_loc, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(features)
        f.close()
    logging.info("Successfully written to csv.")


def get_largest_contour(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image_gray, 150, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_index = 0
    for contour_index in range(len(contours)):
        contour_area = cv2.contourArea(contours[contour_index])
        if contour_area > max_area and len(contours) >= 5:
            max_area = contour_area
            max_index = contour_index
    if len(contours) > 0:
        return contours[max_index]
    else:
        return None


def get_contour_area(contour):
    return cv2.contourArea(contour=contour)


def get_contour_arch_length(contour):
    return cv2.arcLength(contour, closed=True)


def get_eccentricity(contour):
    ellipse = cv2.fitEllipse(contour)
    b = ellipse[1][0]
    a = ellipse[1][1]
    return (1 - (b**2/a**2))**0.5


def get_contour_moments(contour):
    moments = cv2.moments(contour)
    norm_moments = {}
    for key in moments:
        if 'nu' in key:
            norm_moments[key] = moments[key]
    return norm_moments


def get_features_labels_from_csv():
    features = []
    labels = []
    with open(FEATURES_LOCATION, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[-1])
            row = row[:-1]
            features_tuple = []
            for feature in row:
                features_tuple.append(float(feature))
            features.append(features_tuple)
        f.close()
    return features, labels
