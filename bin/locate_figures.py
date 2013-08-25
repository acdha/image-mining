#!/usr/bin/env python

from random import randint
import sys

import cv
import cv2


def process_image(filename):
    source_image = cv2.imread(filename)

    source_gray = cv2.cvtColor(source_image, cv.CV_BGR2GRAY)
    source_gray = cv2.medianBlur(source_gray, 7)
    # source_gray = cv2.blur(source_gray, (3, 3))

    threshold_rc, threshold_image = cv2.threshold(source_gray, 192, 255, cv2.THRESH_BINARY)
    threshold_image = cv2.bitwise_not(threshold_image)

    threshold_image = cv2.erode(threshold_image, cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6)))

    cv2.imshow("Thresh", threshold_image)

    # cv2.namedWindow("Source", cv2.CV_WINDOW_AUTOSIZE)
    # cv2.imshow("Source", source_image)

    def update_canny(thresh, *args):
        edges = cv2.Canny(threshold_image, thresh, thresh * 2, 5)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        output = source_image.copy()

        print "Found %d contours" % len(contours)

        for i, contour in enumerate(contours):
            # length = cv2.arcLength(contours[i], False)
            # area = cv2.contourArea(contours[i], False)
            #
            # if area < 20:
            #     continue
            #
            # print "%0.2f %0.2f" % (length, area)

            color = (randint(0, 255), randint(0, 255), randint(0, 255))

            # poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
            # cv2.polylines(output, contour, True, (128, 255, 128), 2)
            # x, y, w, h = cv2.boundingRect(poly)
            # cv2.rectangle(output, (x, y), (x + w, y + h), (128, 255, 128))

            cv2.drawContours(output, contours, i, color, 1, 8, hierarchy, 0)

        cv2.imshow("Output", output)

    cv2.namedWindow("Output", cv2.CV_WINDOW_AUTOSIZE)
    cv2.createTrackbar("Canny threshold:", "Output", 127, 255, update_canny)
    update_canny(127)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for f in sys.argv[1:]:
        process_image(f)
    cv2.destroyAllWindows()
