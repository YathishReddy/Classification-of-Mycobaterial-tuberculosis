import cv2
import sys
from utils.utils import LARGE_TEST_FILE, PIXEL_CLASSIFIER_TEST_INPUT, FEATURES_ORDER
from classifiers import predict_roi, adaboost, decision_trees, gradient_boosting, multilayer_perceptron
from image_modifiers import denoise_image, sharpen_image, adaptive_threshold, region_of_interest
from data_processor import feature_extractor
import logging


def classify_rectangle(rectangle, image):
    x, y, h, w = rectangle
    classifier_name = sys.argv[1]
    roi = image[y: y + h, x: x + w]
    roi_ = cv2.resize(roi, None, fy=5, fx=5, interpolation=cv2.INTER_LINEAR)
    try:
        features_map = feature_extractor.get_features_from_image(roi_)
        if features_map is None:
            return [2]
        features = []
        for feature in FEATURES_ORDER:
            if feature in features_map:
                features.append(features_map[feature])
            else:
                features.append(features_map["moments"][feature])
        if classifier_name == "nb":
            return predict_roi.classify_region_of_interest(features)
        elif classifier_name == "mlp":
            return multilayer_perceptron.mlp_classifier(features)
        elif classifier_name == "dt":
            return decision_trees.decision_tree_classifier(features)
        elif classifier_name == "gdb":
            return gradient_boosting.gradient_boosting_classifier(features)
        else:
            return adaboost.adaboost_classifier(features)
    except:
        return [2]


def predict_and_bound():
    image = cv2.imread(PIXEL_CLASSIFIER_TEST_INPUT + LARGE_TEST_FILE)
    # denoise -> sharpen -> segment.
    denoised_image = denoise_image(image)
    sharpened_image = sharpen_image(denoised_image)
    segmented_image = adaptive_threshold(sharpened_image)
    cv2.imwrite(PIXEL_CLASSIFIER_TEST_INPUT + "segmented_.png", segmented_image)
    # classify and draw bounded rectangles.
    region_of_interest_ = region_of_interest(segmented_image)
    for rectangle in region_of_interest_:
        x, y, w, h = rectangle
        positive = classify_rectangle(rectangle, image)
        positive = int(positive[0])
        if positive == 1:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif positive == 0:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            logging.info("Received 2.")
    cv2.imwrite(PIXEL_CLASSIFIER_TEST_INPUT + "final_.png", image)

if __name__ == '__main__':
    predict_and_bound()
