#!/usr/bin/env python

import logging

import cv2
import numpy


class ImageRegion(object):
    def __init__(self, x1, y1, x2, y2, poly=None, contour_index=None):
        assert x1 < x2
        assert y1 < y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.poly = poly
        self.contour_index = contour_index

    def __repr__(self):
        return "({0.x1}, {0.y1})-({0.x2}, {0.y2})".format(self)

    @property
    def area(self):
        return (self.y2 - self.y1) * (self.x2 - self.x1)

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def image_slice(self):
        """Return a Python slice suitable for use on an OpenCV image (i.e. numpy 2D array)"""
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def contains(self, other):
        """Returns True if the other ImageRegion is entirely contained by this one"""
        return ((other.x1 >= self.x1) and (other.x2 <= self.x2)
                and (other.y1 >= self.y1) and (other.y2 <= self.y2))

    def overlaps(self, other):
        """Returns True if any part of the other ImageRegion is entirely contained by this one"""

        return (((self.x1 < other.x1 < self.x2) or (self.x1 < other.x2 < self.x2))
                and ((self.y1 < other.y1 < self.y2) or (self.y1 < other.y2 < self.y2)))

    def merge(self, other):
        """Expand this ImageRegion to contain other"""
        self.x1 = min(self.x1, other.x1)
        self.y1 = min(self.y1, other.y1)
        self.x2 = max(self.x2, other.x2)
        self.y2 = max(self.y2, other.y2)

    def as_dict(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


class FigureExtractor(object):
    MORPH_TYPES = {"cross": cv2.MORPH_CROSS,
                   "ellipse": cv2.MORPH_ELLIPSE,
                   "rectangle": cv2.MORPH_RECT}
    MORPH_TYPE_KEYS = sorted(MORPH_TYPES.keys())

    def __init__(self, canny_threshold=0, erosion_element=None, erosion_size=4,
                 dilation_element=None, dilation_size=4,
                 min_area=0.01,
                 min_height=0.1, max_height=0.9,
                 min_width=0.1, max_width=0.9):
        # TODO: reconsider whether we should split to global config + per-image extractor instances

        # TODO: better way to set configuration options & docs
        self.canny_threshold = canny_threshold
        self.erosion_element = self.MORPH_TYPE_KEYS.index(erosion_element)
        self.erosion_size = erosion_size
        self.dilation_element = self.MORPH_TYPE_KEYS.index(dilation_element)
        self.dilation_size = dilation_size

        self.min_area_percentage = min_area
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width

    def find_figures(self, source_image):
        assert source_image is not None, "source_image was None. Perhaps imread() failed?"
        output_image = self.filter_image(source_image)

        contours, hierarchy = self.find_contours(output_image)

        for bbox in self.get_bounding_boxes_from_contours(contours, source_image):
            yield bbox

    def _find_contours_opencv2(self, image):
        return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def _find_contours_opencv3(self, image):
        _, a, b = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return a, b

    if cv2.__version__.startswith('2.'):
        find_contours = _find_contours_opencv2
    else:
        find_contours = _find_contours_opencv3

    def filter_image(self, source_image):
        # TODO: Refactor this into a more reusable filter chain

        output_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        # TODO: make blurring configurable:
        # output_image = cv2.medianBlur(output_image, 5)
        # output_image = cv2.blur(output_image, (3, 3))
        # output_image = cv2.GaussianBlur(output_image, (5, 5))

        # TODO: make thresholding configurable
        # See http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold
        # output_image = cv2.adaptiveThreshold(output_image, 255.0, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_MEAN_C, 15, 5)
        # threshold_rc, output_image = cv2.threshold(output_image, 192, 255, cv2.THRESH_BINARY_INV)

        # Otsu's binarization: see http://bit.ly/194YCPp
        output_image = cv2.GaussianBlur(output_image, (3, 3), 0)
        threshold_rc, output_image = cv2.threshold(output_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.erosion_size > 0:
            element_name = self.MORPH_TYPE_KEYS[self.erosion_element]
            element = self.MORPH_TYPES[element_name]

            structuring_element = cv2.getStructuringElement(element, (self.erosion_size, self.erosion_size))
            output_image = cv2.erode(output_image, structuring_element)

        if self.dilation_size > 0:
            element_name = self.MORPH_TYPE_KEYS[self.dilation_element]
            element = self.MORPH_TYPES[element_name]

            structuring_element = cv2.getStructuringElement(element, (self.dilation_size, self.dilation_size))
            output_image = cv2.dilate(output_image, structuring_element)

        if self.canny_threshold > 0:
            # TODO: Make all of Canny options configurable
            # See http://docs.opencv.org/modules/imgproc/doc/feature_detection.html#canny
            output_image = cv2.Canny(output_image, self.canny_threshold, self.canny_threshold * 3, 12)

        return output_image

    def detect_lines(self, source_image):
        # TODO: Make HoughLinesP a configurable option
        lines = cv2.HoughLinesP(source_image, rho=1, theta=numpy.pi / 180,
                                threshold=160, minLineLength=80, maxLineGap=10)

        # for line in lines[0]:
        #     cv2.line(output_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 4)
        return lines

    def get_bounding_boxes_from_contours(self, contours, source_image):
        # We'll return the boxes ordered largest first to make overlaps easier to see interactively:
        boxes = sorted(self.filter_bounding_boxes(contours, source_image), reverse=True,
                       key=lambda i: i.area)

        # This could be stored in a much more efficient structure but in testing the number
        # of boxes is so small that it doesn't seem worth greater effort:
        boxes = [i for i in boxes if not any(j.contains(i) for j in boxes if j is not i)]

        restart = True
        while restart:
            restart = False
            for i in boxes:
                other_boxes = [j for j in boxes if j is not i]
                for j in other_boxes:
                    if j.overlaps(i):
                        print "\tMerging overlapping extracts: %s %s" % (i, j)
                        i.merge(j)
                        boxes.remove(j)
                        restart = True
                        break

        return boxes

    def filter_bounding_boxes(self, contours, source_image):
        # TODO: confirm that the min area check buys us anything over the bounding box min/max filtering
        min_area = self.min_area_percentage * source_image.size

        # TODO: more robust algorithm for detecting likely scan edge artifacts which can handle cropped scans of large images (e.g. http://dl.wdl.org/107_1_1.png)
        max_height = int(round(self.max_height * source_image.shape[0]))
        max_width = int(round(self.max_width * source_image.shape[1]))
        min_height = int(round(self.min_height * source_image.shape[0]))
        min_width = int(round(self.min_width * source_image.shape[1]))

        logging.info("Contour length & area (area: >%d pixels, box: height >%d, <%d, width >%d, <%d)",
                     min_area, min_height, max_height, min_width, max_width)

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contours[i], False)

            if area < min_area:
                logging.debug("Contour %4d: failed area check", i)
                continue

            poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
            x, y, w, h = cv2.boundingRect(poly)
            bbox = ImageRegion(x, y, x + w, y + h, poly=poly, contour_index=i)

            if w > max_width or w < min_width or h > max_height or h < min_height:
                logging.debug("Contour %4d: failed min/max check: %s", i, bbox)
                continue

            yield bbox
