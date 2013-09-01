#!/usr/bin/env python

import argparse
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


def process_image(filename, output_dir=".", interactive=False,
                  canny_threshold=0, erosion_element=2, erosion_size=4):

    window_name = os.path.splitext(os.path.basename(filename))[0]

    source_image = cv2.imread(filename)
    if source_image is None:
        raise RuntimeError("Unable to load %s" % filename)

    print "Processing %s (%s)" % (window_name, source_image.shape)

    source_gray = cv2.cvtColor(source_image, cv.CV_BGR2GRAY)
    # source_gray = cv2.medianBlur(source_gray, 7)
    # source_gray = cv2.blur(source_gray, (3, 3))

    threshold_rc, threshold_image = cv2.threshold(source_gray, 192, 255, cv2.THRESH_BINARY)
    output_base_image = cv2.bitwise_not(threshold_image)

    def update_output(*args):
        if interactive:
            canny_threshold = cv2.getTrackbarPos("Canny Threshold", window_name)
            erosion_element = cv2.getTrackbarPos("Erosion Element", window_name)
            erosion_size = cv2.getTrackbarPos("Erosion Size", window_name)
        else:
            # FIXME: hack around Python 2 nested scoping craziness:
            canny_threshold=0
            erosion_element=2
            erosion_size=4

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

        # TODO: confirm that the min area check buys us anything over the bounding box min/max filtering
        # TODO: make minimum contour area configurable
        min_area_pct = 0.01
        min_area = min_area_pct * source_image.size

        # TODO: more robust algorithm for detecting likely scan edge artifacts which can handle cropped scans of large images (e.g. http://dl.wdl.org/107_1_1.png)
        max_height = int(round(0.9 * source_image.shape[0]))
        max_width = int(round(0.9 * source_image.shape[1]))
        min_height = int(round(0.1 * source_image.shape[0]))
        min_width = int(round(0.1 * source_image.shape[1]))

        print "Contour length & area (min area: %d pixels, min/max box: height = %d, %d, width = %d, %d)" % (
            min_area, min_height, max_height, min_width, max_width)

        for i, contour in enumerate(contours):
            length = cv2.arcLength(contours[i], False)
            area = cv2.contourArea(contours[i], False)

            if area < min_area:
                continue

            color = (32, 192, 32)

            poly = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, False), False)
            x, y, w, h = cv2.boundingRect(poly)
            bbox = ((x, y), (x + w, y + h))

            if w > max_width or w < min_width or h > max_height or h < min_height:
                print "\t%4d: failed min/max check: %s" % (i, bbox)
                continue

            print "\t%4d: %16.2f%16.2f bounding box=%s" % (i, length, area, bbox)

            extract_name = "%s extract %d" % (window_name, i)
            extracted = source_image[y:y + h, x:x + w]

            if output_dir:
                cv2.imwrite(os.path.join(output_dir, "%s.png" % extract_name), extracted)

            if interactive:
                cv2.imshow(extract_name, extracted)
                cv2.polylines(output, contour, True, tuple(i * 2 for i in color), thickness=3)
                cv2.rectangle(output, bbox[0], bbox[1], color=color)
                cv2.drawContours(output, contours, i, color, hierarchy=hierarchy, maxLevel=0)

        label = []

        if erosion_size:
            label.append("Erosion %s at %d" % (element_name, erosion_size))

        if canny_threshold:
            label.append("Canny at %d" % canny_threshold)

        label.append("%d contours" % len(contours))

        if interactive:
            output = cv2.copyMakeBorder(output, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, (32, 32, 32))
            cv2.putText(output, ", ".join(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (192, 192, 192))
            cv2.imshow(window_name, output)

    if interactive:
        cv2.namedWindow(window_name, cv2.CV_WINDOW_AUTOSIZE)

        cv2.createTrackbar("Erosion Element", window_name,
                           erosion_element, len(MORPH_TYPES) - 1, update_output)
        cv2.createTrackbar("Erosion Size", window_name, erosion_size, 32, update_output)
        cv2.createTrackbar("Canny Threshold", window_name, canny_threshold, 255, update_output)

    update_output()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar="IMAGE_FILE", nargs="+")
    parser.add_argument('--output-directory', default=None, help="Directory to store extracted files")
    parser.add_argument('--interactive', default=False, action="store_true", help="Display visualization windows")
    parser.add_argument('--debug', action="store_true", help="Open debugger for errors")
    args = parser.parse_args()

    if not args.output_directory:
        output_dir = None
    else:
        output_dir = os.path.realpath(args.output_directory)
        if not os.path.isdir(output_dir):
            parser.error("Output directory %s does not exist" % args.output_directory)
        else:
            print "Output will be saved to %s" % output_dir

    if output_dir is None and not args.interactive:
        parser.error("Either use --interactive or specify an output directory to save results!")

    if args.debug:
        try:
            import bpdb as pdb
        except ImportError:
            import pdb

    for f in args.files:
        try:
            process_image(f, output_dir=output_dir, interactive=args.interactive)
        except Exception as exc:
            if args.debug:
                print >>sys.stderr, exc
                pdb.pm()
            raise

    if args.interactive:
        while cv2.waitKey() not in (13, 27):
            continue
    cv2.destroyAllWindows()
