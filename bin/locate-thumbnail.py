#!/usr/bin/env python
# encoding: utf-8
"""
Detect the crop box for a thumbnail inside a larger image

The thumbnail image can be cropped and scaled arbitrarily from the larger image. Rotation and other more
complex transformations should work but may lower accuracy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys

import numpy
import cv2


def match_images(template, source):
    """Return filtered matches from the template and source images"""

    # TODO: Compare non-encumbered options – see http://docs.opencv.org/modules/features2d/doc/features2d.html
    detector = cv2.SURF(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(template, None)
    kp2, desc2 = detector.detectAndCompute(source, None)
    logging.debug('Features: template %d, source %d', len(kp1), len(kp2))

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    kp_pairs = filter_matches(kp1, kp2, raw_matches)

    return kp_pairs


def filter_matches(kp1, kp2, matches, ratio=0.75):
    kp_pairs = []

    for m1, m2 in matches:
        if m1.distance < m2.distance * ratio:
            kp_pairs.append((kp1[m1.queryIdx], kp2[m1.trainIdx]))

    return kp_pairs


def autorotate_image(img, corners):
    corners_x, corners_y = zip(*corners)
    high_corners = (numpy.argmin(corners_y), numpy.argmax(corners_y),
                    numpy.argmin(corners_x), numpy.argmax(corners_x))

    if high_corners == (0, 1, 3, 1):    # 90 degrees
        return numpy.rot90(img, 1)
    elif high_corners == (3, 0, 1, 0):  # 180 degrees
        return cv2.flip(img, -1)
    elif high_corners == (1, 3, 0, 2):  # 270 degrees
        return numpy.rot90(img, 3)
    else:
        # Do nothing for zero-degree rotations
        return img


def reconstruct_thumbnail(thumbnail_image, source_image, kp_pairs, H):
    logging.info("Reconstructing thumbnail from source image")

    thumb_h, thumb_w = thumbnail_image.shape[:2]
    source_h, source_w = source_image.shape[:2]

    corners = numpy.float32([[0, 0], [thumb_w, 0], [thumb_w, thumb_h], [0, thumb_h]])
    corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

    logging.info("Thumbnail bounds within source image: %s", corners.tolist())

    corners_x, corners_y = zip(*corners)

    new_thumb = source_image[min(corners_y):max(corners_y), min(corners_x):max(corners_x)]

    new_thumb = autorotate_image(new_thumb, corners)

    new_thumb_h, new_thumb_w = new_thumb.shape[:2]
    if new_thumb_h > thumb_h or new_thumb_w > thumb_w:
        if thumb_h > thumb_w:
            scale = thumb_h / new_thumb_h
        else:
            scale = thumb_w / new_thumb_w

        new_dims = (int(round(new_thumb_w * scale)), int(round(new_thumb_h * scale)))

        logging.info("Resizing from %s to %s", new_thumb.shape[:2], new_dims)

        new_thumb = cv2.resize(new_thumb, new_dims, interpolation=cv2.INTER_AREA)

    print("Master dimensions: {}x{}".format(*source_image.shape))
    print("Thumbnail dimensions: {}x{}".format(*thumbnail_image.shape))
    print("Reconstructed thumb dimensions: {}x{}".format(*new_thumb.shape))

    return new_thumb, corners


def visualize_matches(source_image, original_thumbnail, reconstructed_thumbnail, corners, kp_pairs, mask):
    thumb_h, thumb_w = original_thumbnail.shape[:2]
    source_h, source_w = source_image.shape[:2]

    # Create a new image for the visualization:
    vis = numpy.zeros((max(thumb_h, source_h), thumb_w + source_w, source_image.shape[2]), numpy.uint8)
    # Draw the original images adjacent to each other:
    vis[:thumb_h, :thumb_w] = original_thumbnail
    vis[:source_h, thumb_w:thumb_w+source_w] = source_image

    if reconstructed_thumbnail is not None:
        # Display the reconstructed thumbnail just below the original thumbnail:
        reconstructed_h, reconstructed_w = reconstructed_thumbnail.shape[:2]
        vis[thumb_h:thumb_h + reconstructed_h, :reconstructed_w] = reconstructed_thumbnail

    if corners is not None:
        # Highlight our bounding box on the source image:
        cv2.polylines(vis, [corners + (thumb_w, 0)], True, (255, 255, 255))

    thumb_points = numpy.int32([kpp[0].pt for kpp in kp_pairs])
    source_points = numpy.int32([kpp[1].pt for kpp in kp_pairs]) + (thumb_w, 0)

    # Points which fit the model will be marked in green:
    inlier_color = (0, 255, 0)
    # … while those which do not will be marked in red:
    outlier_color = (0, 0, 255)
    # Connecting lines will be less intense green:
    line_color = (0, 192, 0)

    if mask is None:
        mask = numpy.zeros(len(thumb_points))

    for (x1, y1), (x2, y2), inlier in zip(thumb_points, source_points, mask):
        if inlier:
            cv2.line(vis, (x1, y1), (x2, y2), line_color)
            cv2.circle(vis, (x1, y1), 2, inlier_color, -1)
            cv2.circle(vis, (x2, y2), 2, inlier_color, -1)
        else:
            cv2.circle(vis, (x1, y1), 2, outlier_color, -1)
            cv2.circle(vis, (x2, y2), 2, outlier_color, -1)

    return vis


def find_homography(kp_pairs):
    mkp1, mkp2 = zip(*kp_pairs)

    p1 = numpy.float32([kp.pt for kp in mkp1])
    p2 = numpy.float32([kp.pt for kp in mkp2])

    assert len(kp_pairs) >= 4

    logging.debug('finding homography')
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
    logging.info('%d inliers, %d matched features', numpy.sum(mask), len(mask))
    return H, mask


def load_image(filename):
    if not os.path.exists(filename):
        raise RuntimeError("%s does not exist" % filename)

    img = cv2.imread(filename)

    if img is None:
        raise RuntimeError('Failed to load %s' % filename)

    return img


def locate_thumbnail(thumbnail_filename, source_filename, display=False, save_visualization=False):
    thumbnail_image = load_image(thumbnail_filename)
    source_image = load_image(source_filename)

    logging.info("Attempting to locate %s within %s", thumbnail_filename, source_filename)
    kp_pairs = match_images(thumbnail_image, source_image)

    if len(kp_pairs) >= 4:
        title = "Found %d matches" % len(kp_pairs)
        logging.info(title)

        H, mask = find_homography(kp_pairs)

        new_thumbnail, corners = reconstruct_thumbnail(thumbnail_image, source_image, kp_pairs, H)

        new_filename = "%s.reconstructed%s" % os.path.splitext(thumbnail_filename)
        cv2.imwrite(new_filename, new_thumbnail)
        logging.info("Saved reconstructed thumbnail %s", new_filename)
    else:
        print("Found only %d matches; skipping reconstruction" % len(kp_pairs), file=sys.stderr)
        new_thumbnail = corners = H = mask = None

    if display or save_visualization:
        vis_image = visualize_matches(source_image, thumbnail_image, new_thumbnail, corners, kp_pairs, mask)

    if save_visualization:
        vis_filename = "%s.visualized%s" % os.path.splitext(thumbnail_filename)
        cv2.imwrite(vis_filename, vis_image)
        logging.info("Saved match visualization %s", vis_filename)

    if display:
        cv2.imshow(title, vis_image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(funcName)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar="THUMBNAIL MASTER", nargs="+")
    parser.add_argument('--save-visualization', action="store_true", help="Save match visualization")
    parser.add_argument('--display', action="store_true", help="Display match visualization")
    parser.add_argument('--debug', action="store_true", help="Open debugger for errors")
    args = parser.parse_args()

    if len(args.files) % 2 != 0:
        parser.error("Files must be provided in thumbnail and master pairs")

    if args.debug:
        try:
            import bpdb as pdb
        except ImportError:
            import pdb

    for i in xrange(0, len(args.files), 2):
        thumbnail = args.files[i]
        source = args.files[i + 1]
        try:
            locate_thumbnail(thumbnail, source, display=args.display,
                             save_visualization=args.save_visualization)
        except Exception as e:
            logging.error("Error processing %s %s: %s", thumbnail, source, e)
            if args.debug:
                pdb.pm()


if __name__ == '__main__':
    main()