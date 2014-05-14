#!/usr/bin/env python
# encoding: utf-8
"""
Detect the crop box for a thumbnail inside a larger image

The thumbnail image can be cropped and scaled arbitrarily from the larger image. Rotation and other more
complex transformations should work but may lower accuracy.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import json
import logging
import os
import sys

import cv
import cv2
import numpy
from image_mining.utils import open_image


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

    # n.b. numpy rot90 rotates 90° counter-clockwise but our terminology is clockwise
    #      so the rotations below aren't actually flippy:

    print(corners_x, corners_y)

    if (((min(corners_x[0], corners_x[1]) > max(corners_x[2], corners_x[3]))
         and min(corners_y[1], corners_y[2]) > max(corners_y[0], corners_y[3]))):
        return 270, numpy.rot90(img)
    elif min(corners_x[2], corners_x[3]) > max(corners_x[0], corners_x[1]):
        return 90, numpy.rot90(img, 3)
    elif min(corners_x[0], corners_x[3]) > max(corners_x[1], corners_x[2]):
        return 180, cv2.flip(img, -1)
    else:
        return 0, img


def fit_image_within(img, max_height, max_width):
    current_h, current_w = img.shape[:2]

    # Confirm that we need to do anything:
    if current_h <= max_height and current_w <= max_width:
        return img

    if current_h > current_w:
        scale = max_height / current_h
    else:
        scale = max_width / current_w

    new_dims = (int(round(current_w * scale)), int(round(current_h * scale)))

    # Note the flip from numpy's .shape to opencv's (x, y) format:
    logging.info('Resizing from %s to %s', (current_w, current_h), new_dims)

    return cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)


def get_scaled_corners(thumbnail_image, source_image, full_source_image, kp_pairs, H):
    thumb_h, thumb_w = thumbnail_image.shape[:2]

    corners = numpy.float32([[0, 0], [thumb_w, 0], [thumb_w, thumb_h], [0, thumb_h]])
    corners = numpy.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2))

    # It's possible for rounding errors to produce values which are slightly outside of the image dimensions
    # so we'll clamp the boundaries within the source image: https://github.com/acdha/image-mining/issues/5
    source_h, source_w = source_image.shape[:2]

    # Transpose the array so we can operate on it *in-place* to clamp values:
    corners_x, corners_y = corners.T
    numpy.clip(corners_x, 0.0, source_w, out=corners_x)
    numpy.clip(corners_y, 0.0, source_h, out=corners_y)

    corners = corners.tolist()

    logging.info("Thumbnail bounds within analyzed image: %s", corners)

    if full_source_image is not None and full_source_image is not source_image:
        scale_y = full_source_image.shape[0] / source_image.shape[0]
        scale_x = full_source_image.shape[1] / source_image.shape[1]

        corners = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in corners]

        logging.info("Thumbnail bounds within full-size source image: %s", corners)

    return corners


def adjust_crop_aspect_ratio(cropbox, target_aspect_ratio, original_height=0, original_width=0,
                             max_height=0, max_width=0):

    new_crop_y, new_crop_x = cropbox
    new_crop_height = (new_crop_y[1] - new_crop_y[0])
    new_crop_width = (new_crop_x[1] - new_crop_x[0])
    new_aspect_ratio = new_crop_height / new_crop_width

    if abs(target_aspect_ratio - new_aspect_ratio) < 0.001:
        return cropbox

    logging.info('Adjusting reconstruction to match original %0.4f aspect ratio', target_aspect_ratio)

    assert original_height < new_crop_height
    assert original_width < new_crop_width

    # The basic idea is that we'll adjust the crop's short axis up or down to match the input aspect
    # ratio. To avoid shifting the crop too much we'll attempt to evenly move both sides as long as
    # that won't hit the image boundaries:

    if new_aspect_ratio > 1.0:
        scale = new_crop_width / original_width
    else:
        scale = new_crop_height / original_height

    logging.info('Original crop box: %r (%0.4f)', cropbox, new_crop_height / new_crop_width)
    logging.info('Reconstructed image is %0.2f%% of the original', scale * 100)

    delta_y = round(original_height * scale) - new_crop_height
    delta_x = round(original_width * scale) - new_crop_width

    logging.info('Crop box needs to change by: %0.1f x, %0.1f y', delta_x, delta_y)

    if delta_y != 0:
        new_crop_y = clamp_values(delta=delta_y, max_value=max_height, *cropbox[0])

    if delta_x != 0:
        new_crop_x = clamp_values(delta=delta_x, max_value=max_width, *cropbox[1])

    cropbox = (new_crop_y, new_crop_x)

    logging.info('Updated crop box: %r (%0.4f)', cropbox,
                 (new_crop_y[1] - new_crop_y[0]) / (new_crop_x[1] - new_crop_x[0]))

    return cropbox


def clamp_values(low_value, high_value, delta, min_value=0, max_value=0):
    if delta == 0.0:
        return low_value, high_value

    top_pad = bottom_pad = delta / 2

    if delta > 0:
        # We'll shift the box to avoid hitting an image edge:
        top_pad = max(0, top_pad)
        bottom_pad = delta - top_pad

    low_value = int(round(low_value - top_pad))

    if low_value < min_value:
        logging.warning('Clamping crop to %f instead of %f', min_value, low_value)
        bottom_pad += min_value - low_value
        low_value = min_value

    high_value = int(round(high_value + bottom_pad))

    if high_value > max_value:
        logging.warning('Clamping crop to %f instead of %f', max_value, high_value)
        high_value = max_value

    return low_value, high_value


def reconstruct_thumbnail(thumbnail_image, source_image, corners, downsize_reconstruction=False,
                          max_aspect_ratio_delta=0.1, match_aspect_ratio=False):
    logging.info("Reconstructing thumbnail from source image")

    thumb_h, thumb_w = thumbnail_image.shape[:2]
    source_h, source_w = source_image.shape[:2]

    old_aspect_ratio = thumb_h / thumb_w

    corners_x, corners_y = zip(*corners)
    new_thumb_crop = [(min(corners_y), max(corners_y)),
                      (min(corners_x), max(corners_x))]

    if match_aspect_ratio:
        new_thumb_crop = adjust_crop_aspect_ratio(new_thumb_crop, old_aspect_ratio,
                                                  original_height=thumb_h,
                                                  original_width=thumb_w,
                                                  max_height=source_h, max_width=source_w)

    new_thumb = source_image[slice(*new_thumb_crop[0]), slice(*new_thumb_crop[1])]

    new_thumb_rotation, new_thumb = autorotate_image(new_thumb, corners)
    logging.info('Detected image rotation: %d°', new_thumb_rotation)

    if match_aspect_ratio and new_thumb_rotation not in (0, 180):
        raise NotImplementedError('FIXME: refactor autorotation to work with aspect ratio matching!')

    new_thumb_h, new_thumb_w = new_thumb.shape[:2]

    if downsize_reconstruction and (new_thumb_h > thumb_h or new_thumb_w > thumb_w):
        new_thumb = fit_image_within(new_thumb, thumb_h, thumb_w)

    new_aspect_ratio = new_thumb.shape[0] / new_thumb.shape[1]
    logging.info('Master dimensions: width=%s, height=%s', source_image.shape[1], source_image.shape[0])
    logging.info('Thumbnail dimensions: width=%s, height=%s (aspect ratio: %0.4f)',
                 thumbnail_image.shape[1], thumbnail_image.shape[0],
                 old_aspect_ratio)
    logging.info('Reconstructed thumb dimensions: width=%s, height=%s (rotation=%d°, aspect ratio: %0.4f)',
                 new_thumb.shape[1], new_thumb.shape[0],
                 new_thumb_rotation, new_aspect_ratio)

    if match_aspect_ratio:
        scale = thumbnail_image.shape[0] / new_thumb.shape[0]
        if thumbnail_image.shape[:2] != tuple(int(round(i * scale)) for i in new_thumb.shape[:2]):
            raise RuntimeError('Unable to match aspect ratios: %0.4f != %0.4f' % (old_aspect_ratio,
                                                                                  new_aspect_ratio))

    if abs(old_aspect_ratio - new_aspect_ratio) > max_aspect_ratio_delta:
        raise RuntimeError('Aspect ratios are significantly different – reconstruction likely failed!')

    if (new_thumb_h <= thumb_h) or (new_thumb_w <= thumb_w):
        raise RuntimeError("Reconstructed thumbnail wasn't larger than the original!")

    return new_thumb, new_thumb_crop, new_thumb_rotation


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
        reconstructed_thumbnail = fit_image_within(reconstructed_thumbnail, thumb_h, thumb_w)
        reconstructed_h, reconstructed_w = reconstructed_thumbnail.shape[:2]
        vis[thumb_h:thumb_h + reconstructed_h, :reconstructed_w] = reconstructed_thumbnail

    if corners is not None:
        # Highlight our bounding box on the source image:
        cv2.polylines(vis, [numpy.int32(corners) + (thumb_w, 0)], True, (255, 255, 255))

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


def locate_thumbnail(thumbnail_filename, source_filename, display=False, save_visualization=False,
                     save_reconstruction=False, reconstruction_format="jpg",
                     max_aspect_ratio_delta=0.1, match_aspect_ratio=False,
                     minimum_matches=10,
                     json_output_filename=None, max_master_edge=4096, max_output_edge=2048):
    thumbnail_basename, thumbnail_image = open_image(thumbnail_filename)
    source_basename, source_image = open_image(source_filename)

    if (((source_image.shape[0] <= thumbnail_image.shape[0])
         or (source_image.shape[1] <= thumbnail_image.shape[1]))):
        raise RuntimeError("Master file wasn't larger than the thumbnail: %r vs %r" % (source_image.shape,
                                                                                       thumbnail_image.shape))

    logging.info("Attempting to locate %s within %s", thumbnail_filename, source_filename)

    full_source_image = source_image
    if max_master_edge and any(i for i in source_image.shape if i > max_master_edge):
        logging.info("Resizing master to fit within %d pixels", max_master_edge)
        source_image = fit_image_within(source_image, max_master_edge, max_master_edge)

    logging.info('Finding common features')
    kp_pairs = match_images(thumbnail_image, source_image)

    if len(kp_pairs) >= minimum_matches:
        title = "Found %d matches" % len(kp_pairs)
        logging.info(title)

        H, mask = find_homography(kp_pairs)

        corners = get_scaled_corners(thumbnail_image, source_image, full_source_image, kp_pairs, H)

        new_thumbnail, corners, rotation = reconstruct_thumbnail(thumbnail_image, full_source_image, corners,
                                                                 match_aspect_ratio=match_aspect_ratio,
                                                                 max_aspect_ratio_delta=max_aspect_ratio_delta)

        if json_output_filename:
            with open(json_output_filename, mode='wb') as json_file:
                json.dump({
                    "master": {
                        "source": source_filename,
                        "dimensions": {
                            "height": full_source_image.shape[0],
                            "width": full_source_image.shape[1],
                        }
                    },
                    "thumbnail": {
                        "source": thumbnail_filename,
                        "dimensions": {
                            "height": thumbnail_image.shape[0],
                            "width": thumbnail_image.shape[1],
                        }
                    },
                    "bounding_box": {
                        "height": corners[0][1] - corners[0][0],
                        "width": corners[1][1] - corners[1][0],
                        "x": corners[1][0],
                        "y": corners[0][0],
                    },
                    "rotation_degrees": rotation
                }, json_file, indent=4)

        if save_reconstruction:
            new_filename = "%s.reconstructed.%s" % (thumbnail_basename, reconstruction_format)

            new_thumb_img = fit_image_within(new_thumbnail, max_output_edge, max_output_edge)
            cv2.imwrite(new_filename, new_thumb_img)
            logging.info("Saved reconstructed %s thumbnail %s", new_thumb_img.shape[:2], new_filename)
    else:
        logging.warning("Found only %d matches; skipping reconstruction", len(kp_pairs))
        title = "MATCH FAILED: %d pairs" % len(kp_pairs)
        new_thumbnail = corners = H = mask = None

    if display or save_visualization:
        vis_image = visualize_matches(source_image, thumbnail_image, new_thumbnail, corners, kp_pairs, mask)

    if save_visualization:
        vis_filename = "%s.visualized%s" % os.path.splitext(thumbnail_filename)
        cv2.imwrite(vis_filename, vis_image)
        logging.info("Saved match visualization %s", vis_filename)

    if display:
        # This may or may not exist depending on whether OpenCV was compiled using the QT backend:
        window_flags = getattr(cv, 'CV_WINDOW_NORMAL', cv.CV_WINDOW_AUTOSIZE)
        window_title = '%s - %s' % (thumbnail_basename, title)
        cv2.namedWindow(window_title, flags=window_flags)
        cv2.imshow(window_title, vis_image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(funcName)s: %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar="THUMBNAIL MASTER", nargs="+")
    parser.add_argument('--save-visualization', action="store_true", help="Save match visualization")
    parser.add_argument('--save-thumbnail', action="store_true",
                        help="Save reconstructed thumbnail at full size")
    parser.add_argument('--save-json', action="store_true",
                        help="Save JSON file with thumbnail crop information")
    parser.add_argument('--thumbnail-format', default='jpg',
                        help='Format for reconstructed thumbnails (png or default %(default)s)')
    parser.add_argument('--fit-master-within', type=int, default=8192,
                        help="Resize master so the largest edge is below the specified value "
                             "(faster but possibly less accurate)")
    parser.add_argument('--fit-output-within', type=int, default=2048,
                        help="Resize output so the largest edge is below the specified value")
    parser.add_argument('--minimum-matches', type=int, default=20,
                        help='Require at least this many features for a match (default %(default)s)')
    parser.add_argument('--max-aspect-ratio-delta', type=float, default=0.1,
                        help='Raise an error if the reconstructed image\'s aspect ratio differs by more than '
                             'this percentage default %(default)s)')
    parser.add_argument('--match-aspect-ratio', action='store_true',
                        help='Adjust the reconstructed crop box to exactly match the original thumbnail')
    parser.add_argument('--display', action="store_true", help="Display match visualization")
    parser.add_argument('--debug', action="store_true", help="Open debugger for errors")
    args = parser.parse_args()

    if len(args.files) % 2 != 0:
        parser.error("Files must be provided in thumbnail and master pairs")

    if args.thumbnail_format not in ('jpg', 'png'):
        parser.error('Thumbnail format must be either jpg or png')

    if args.debug:
        import pdb

    for i in xrange(0, len(args.files), 2):
        thumbnail = args.files[i]
        source = args.files[i + 1]

        if args.save_json:
            json_output_filename = '%s.json' % os.path.splitext(thumbnail)[0]
        else:
            json_output_filename = None

        try:
            locate_thumbnail(thumbnail, source, display=args.display,
                             save_reconstruction=args.save_thumbnail,
                             reconstruction_format=args.thumbnail_format,
                             save_visualization=args.save_visualization,
                             json_output_filename=json_output_filename,
                             max_master_edge=args.fit_master_within,
                             max_output_edge=args.fit_output_within,
                             max_aspect_ratio_delta=args.max_aspect_ratio_delta,
                             match_aspect_ratio=args.match_aspect_ratio,
                             minimum_matches=args.minimum_matches)
        except Exception as e:
            logging.error("Error processing %s %s: %s", thumbnail, source, e)
            if args.debug:
                pdb.post_mortem()
            sys.exit(1)


if __name__ == '__main__':
    main()
