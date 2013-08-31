#!/usr/bin/env python

from random import randint
import sys

import cv
import cv2

MORPH_TYPES = {
    "cross": cv2.MORPH_CROSS,
    "ellipse": cv2.MORPH_ELLIPSE,
    "rectangle": cv2.MORPH_RECT,
}
MORPH_TYPE_KEYS = sorted(MORPH_TYPES.keys())


def process_image(filename):
    source_image = cv2.imread(filename)

    # Remove once OpenCV unbreaks window autoresizing:
    # source_image = cv2.resize(source_image, (800, 600))

    source_gray = cv2.cvtColor(source_image, cv.CV_BGR2GRAY)
    source_gray = cv2.medianBlur(source_gray, 7)
    # source_gray = cv2.blur(source_gray, (3, 3))

    threshold_rc, threshold_image = cv2.threshold(source_gray, 192, 255, cv2.THRESH_BINARY)
    output_base_image = cv2.bitwise_not(threshold_image)

    def update_output(*args):
        canny_threshold = cv2.getTrackbarPos("Canny Threshold", "Output")
        erosion_element = cv2.getTrackbarPos("Erosion Element", "Output")
        erosion_size = cv2.getTrackbarPos("Erosion Size", "Output")

        element_name = MORPH_TYPE_KEYS[erosion_element]
        element = MORPH_TYPES[element_name]

        structuring_element = cv2.getStructuringElement(element, (2 * erosion_size + 1,
                                                                  2 * erosion_size + 1))
        print "Erosion type %s size %d" % (element_name, erosion_size)
        output_image = cv2.erode(output_base_image, structuring_element)

        if canny_threshold == 0:
            edges = output_image
        else:
            print "Running Canny at threshold %d" % canny_threshold
            edges = cv2.Canny(output_image, canny_threshold, canny_threshold * 3, 12)

        print "Finding contours...",
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print " found %d" % len(contours)

        # This allows us to draw in color below:
        output = cv2.cvtColor(output_image, cv.CV_GRAY2RGB)

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

        label = ["Erosion %s at %d" % (element_name, erosion_size)]

        if canny_threshold:
            label.append("Canny at %d" % canny_threshold)

        label.append("%d contours" % len(contours))

        output = cv2.copyMakeBorder(output, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, (32, 32, 32))
        cv2.putText(output, ", ".join(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (192, 192, 192))

        cv2.imshow("Output", output)

    cv2.namedWindow("Output", cv2.CV_WINDOW_AUTOSIZE)

    cv2.createTrackbar("Erosion Element", "Output",
                       2, len(MORPH_TYPES) - 1, update_output)
    cv2.createTrackbar("Erosion Size", "Output", 4, 32, update_output)
    cv2.createTrackbar("Canny Threshold", "Output", 0, 255, update_output)

    update_output()

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        import bpdb as pdb
    except ImportError:
        import pdb

    for f in sys.argv[1:]:
        try:
            process_image(f)
        except:
            pdb.pm()
            raise

    cv2.destroyAllWindows()
