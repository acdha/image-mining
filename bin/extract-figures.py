#!/usr/bin/env python

import argparse
import json
import os
import sys

import cv2
import numpy

from image_mining.figure_extraction import FigureExtractor
from image_mining.utils import open_image


def display_images(extractor, files):
    window_name = "Controls"

    images = []
    for f in files:
        print "Loading %s" % f

        try:
            images.append(open_image(f))
        except StandardError as exc:
            print >>sys.stderr, exc
            continue

    def update_display(*args):
        extractor.canny_threshold = cv2.getTrackbarPos("Canny Threshold", window_name)
        extractor.erosion_element = cv2.getTrackbarPos("Erosion Element", window_name)
        extractor.erosion_size = cv2.getTrackbarPos("Erosion Size", window_name)
        extractor.dilation_element = cv2.getTrackbarPos("Dilation Element", window_name)
        extractor.dilation_size = cv2.getTrackbarPos("Dilation Size", window_name)

        # TODO: tame configuration hideousness:
        labels = ["Canny Threshold: %s" % extractor.canny_threshold,
                  "Erosion Element: %s" % FigureExtractor.MORPH_TYPE_KEYS[extractor.erosion_element],
                  "Erosion Size: %s" % extractor.erosion_size,
                  "Dilation Element: %s" % FigureExtractor.MORPH_TYPE_KEYS[extractor.dilation_element],
                  "Dilation Size: %s" % extractor.dilation_size]

        labels_img = numpy.zeros((30 * (len(labels) + 1), 600, 3), numpy.uint8)
        for i, label in enumerate(labels, 1):
            cv2.putText(labels_img, label, (0, i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (192, 192, 192))
        cv2.imshow("Controls", labels_img)

        print "Settings:\n\t", "\n\t".join(labels)
        print

        for name, image in images:
            filtered_image = extractor.filter_image(image)
            contours, hierarchy = extractor.find_contours(filtered_image)

            # The filtered image will be heavily processed down to 1-bit depth. We'll convert it to RGB
            # so we can display the effects of the filters with full-color overlays for detected figures:
            output = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)

            print "Processing %s" % name

            for bbox in extractor.get_bounding_boxes_from_contours(contours, filtered_image):
                print "\tExtract: %s" % bbox
                output[bbox.image_slice] = image[bbox.image_slice]

                cv2.polylines(output, bbox.poly, True, (32, 192, 32), thickness=3)
                cv2.drawContours(output, contours, bbox.contour_index, (32, 192, 32), hierarchy=hierarchy, maxLevel=0)

                cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color=(32, 192, 192))

            cv2.imshow(name, output)

    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 600, 340)

    cv2.createTrackbar("Canny Threshold", window_name, extractor.canny_threshold, 255, update_display)
    cv2.createTrackbar("Erosion Element", window_name, extractor.erosion_element, len(extractor.MORPH_TYPES) - 1, update_display)
    cv2.createTrackbar("Erosion Size", window_name, extractor.erosion_size, 64, update_display)
    cv2.createTrackbar("Dilation Element", window_name, extractor.dilation_element, len(extractor.MORPH_TYPES) - 1, update_display)
    cv2.createTrackbar("Dilation Size", window_name, extractor.dilation_size, 64, update_display)

    update_display()

    if args.interactive:
        while cv2.waitKey() not in (13, 27):
            continue
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action="store_true", help="Open debugger for errors")

    parser.add_argument('files', metavar="IMAGE_FILE", nargs="+")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--interactive', default=False, action="store_true", help="Display visualization windows")
    mode_group.add_argument('--output-directory', default=None, help="Directory to store extracted files")

    parser.add_argument('--save-json', action="store_true", help="Save bounding boxes as JSON files along with extracts")

    extraction_params = parser.add_argument_group("Extraction Parameters")
    extraction_params.add_argument('--canny-threshold', type=int, default=0, help="Canny edge detection threshold (%(type)s, default=%(default)s, 0 to disable)")

    extraction_params.add_argument('--erosion-element', default="rectangle", choices=FigureExtractor.MORPH_TYPE_KEYS, help="Erosion Element (default: %(default)s)")
    extraction_params.add_argument('--erosion-size', type=int, default=0, help="Erosion Size (%(type)s, default=%(default)s, 0 to disable)")

    extraction_params.add_argument('--dilation-element', default="rectangle", choices=FigureExtractor.MORPH_TYPE_KEYS, help="Dilation Element (default: %(default)s)")
    extraction_params.add_argument('--dilation-size', type=int, default=0, help="Dilation Size (%(type)s, default=%(default)s, 0 to disable)")

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

    # FIXME: we should have a way to enumerate this from FigureExtractor and feed argparse that way:
    param_names = [action.dest for action in extraction_params._group_actions]
    params = {k: v for (k, v) in args._get_kwargs() if k in param_names}

    try:
        extractor = FigureExtractor(**params)

        if args.interactive:
            display_images(extractor, args.files)
        else:
            for f in args.files:
                try:
                    base_name, source_image = open_image(f)
                except StandardError as exc:
                    print >>sys.stderr, exc
                    continue

                output_base = os.path.join(output_dir, base_name)

                print "Processing %s" % f

                boxes = []

                for i, bbox in enumerate(extractor.find_figures(source_image), 1):
                    extracted = source_image[bbox.image_slice]
                    extract_filename = os.path.join(output_dir, "%s-%d.jpg" % (output_base, i))
                    print "\tSaving %s" % extract_filename
                    cv2.imwrite(extract_filename, extracted)

                    boxes.append(bbox.as_dict())

                if args.save_json and boxes:
                    json_data = {"source_image": {"filename": f,
                                                  "dimensions": {"width": source_image.shape[1],
                                                                 "height": source_image.shape[0]}},
                                 "regions": boxes}

                    json_filename = os.path.join(output_dir, "%s.json" % output_base)
                    with open(json_filename, "wb") as json_f:
                        json.dump(json_data, json_f, allow_nan=False)
                    print "\tSaved extract information to %s" % json_filename

    except Exception as exc:
        if args.debug:
            print >>sys.stderr, exc
            pdb.pm()
        raise
