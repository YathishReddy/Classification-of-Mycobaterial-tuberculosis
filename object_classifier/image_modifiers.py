import cv2
import numpy as np

from utils.utils import PIXEL_CLASSIFIER_TEST_INPUT


def denoise_image(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    cv2.imwrite(PIXEL_CLASSIFIER_TEST_INPUT + "denoised_image_.png", denoised_image)
    return denoised_image


def sharpen_image(image):
    kernel = np.array([ [-1,-1,-1], [-1, 9, -1], [-1, -1, -1] ])
    res = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(PIXEL_CLASSIFIER_TEST_INPUT + "sharpened_.jpg", res)
    return res


def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    kernel_opening = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening)
    kernel_closing = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing)
    return closing


def region_of_interest(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounded_rectangles = []
    for i in xrange(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        nx = x-5 if x > 5 else 0
        ny = y-5 if y > 5 else 0
        nw = w+10 if nx + w + 10 < image.shape[1] else image.shape[1]-1
        nh = h+10 if ny + h + 10 < image.shape[0] else image.shape[0]-1
        rect = (nx, ny, nw,  nh)
        bounded_rectangles.append(rect)
        # print x,y,w,h
        cv2.rectangle(image, (nx, ny), (nx+nw, ny+nh), (255, 255, 255), 2)
    cv2.imwrite(PIXEL_CLASSIFIER_TEST_INPUT + "rectangle_.png", image)
    return bounded_rectangles
