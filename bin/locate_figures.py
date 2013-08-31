#!/usr/bin/env python

import os
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
    window_name = os.path.basename(filename)

    source_image = cv2.imread(filename)

    print "%s: %s" % (window_name, source_image.shape)

    # Remove once OpenCV unbreaks window autoresizing:
    # source_image = cv2.resize(source_image, None, fx=0.5, fy=0.5)

    source_gray = cv2.cvtColor(source_image, cv.CV_BGR2GRAY)
    source_gray = cv2.medianBlur(source_gray, 7)
    # source_gray = cv2.blur(source_gray, (3, 3))

    threshold_rc, threshold_image = cv2.threshold(source_gray, 192, 255, cv2.THRESH_BINARY)
    output_base_image = cv2.bitwise_not(threshold_image)

    def update_output(*args):
        canny_threshold = cv2.getTrackbarPos("Canny Threshold", window_name)
        erosion_element = cv2.getTrackbarPos("Erosion Element", window_name)
        erosion_size = cv2.getTrackbarPos("Erosion Size", window_name)

        output_image = output_base_image.copy()

        if erosion_size > 0:
            element_name = MORPH_TYPE_KEYS[erosion_element]
            element = MORPH_TYPES[element_name]

            structuring_element = cv2.getStructuringElement(element, (2 * erosion_size + 1,
                                                                      2 * erosion_size + 1))
            print "Erosion type %s size %d" % (element_name, erosion_size)
            output_image = cv2.erode(output_image, structuring_element)

        if canny_threshold > 0:
            print "Running Canny at threshold %d" % canny_threshold
            output_image = cv2.Canny(output_image, canny_threshold, canny_threshold * 3, 12)

        print "Finding contours...",
        contours, hierarchy = cv2.findContours(output_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print " found %d" % len(contours)

        output = source_image.copy()

        if False:
            lines = cv2.HoughLinesP(output_base_image, rho=1, theta=cv.CV_PI / 180,
                                    threshold=160, minLineLength=80, maxLineGap=10)

            for line in lines[0]:
                cv2.line(output, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 4)

        # TODO: make minimum contour area configurable:
        min_area_pct = 0.01
        min_area = min_area_pct * source_image.size
        print "Contour length & area (exclusion threshold: %0.2f%% == %d pixels)" % (min_area_pct, min_area)

        for i, contour in enumerate(contours):
            length = cv2.arcLength(contours[i], False)
            area = cv2.contourArea(contours[i], False)

            if area < min_area:
                continue

            print "\t%4d: %16.2f%16.2f" % (i, length, area)

            color = (32, 192, 32)

            poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
            cv2.polylines(output, contour, True, tuple(i * 2 for i in color), thickness=3)

            x, y, w, h = cv2.boundingRect(poly)
            cv2.rectangle(output, (x, y), (x + w, y + h), color=color)

            cv2.drawContours(output, contours, i, color, hierarchy=hierarchy, maxLevel=0)

        label = []

        if erosion_size:
            label.append("Erosion %s at %d" % (element_name, erosion_size))

        if canny_threshold:
            label.append("Canny at %d" % canny_threshold)

        label.append("%d contours" % len(contours))

        output = cv2.copyMakeBorder(output, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, (32, 32, 32))
        cv2.putText(output, ", ".join(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (192, 192, 192))

        cv2.imshow(window_name, output)

    cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)

    cv2.createTrackbar("Erosion Element", window_name,
                       2, len(MORPH_TYPES) - 1, update_output)
    cv2.createTrackbar("Erosion Size", window_name, 4, 32, update_output)
    cv2.createTrackbar("Canny Threshold", window_name, 0, 255, update_output)

    update_output()

if __name__ == "__main__":
    try:
        import bpdb as pdb
    except ImportError:
        import pdb

    for f in sys.argv[1:]:
        try:
            process_image(f)
        except Exception as exc:
            print >>sys.stderr, exc
            pdb.pm()
            raise

    while cv2.waitKey() not in (13, 27):
        continue
    cv2.destroyAllWindows()
