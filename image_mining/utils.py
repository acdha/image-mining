# encoding: utf-8
from __future__ import absolute_import, unicode_literals, print_function

from urllib import urlopen
from urlparse import urlparse
import os

import cv2
import numpy


def open_image(file_or_url):
    """Load an OpenCV image from a filename or URL

    Returns a base_name, image tuple containing a processed name suitable for naming output files
    """

    if file_or_url.startswith("http"):
        source_image = open_image_from_url(file_or_url, cv2_img_flag=cv2.IMREAD_COLOR)

        url_p = urlparse(file_or_url)

        base_name = os.path.splitext(os.path.basename(url_p.path))[0]
    else:
        if not os.path.exists(file_or_url):
            raise IOError("%s does not exist" % file_or_url)

        base_name = os.path.splitext(os.path.basename(file_or_url))[0]

        source_image = cv2.imread(file_or_url, flags=cv2.IMREAD_COLOR)

    if source_image is None:
        raise RuntimeError("%s could not be decoded as an image" % file_or_url)

    return base_name, source_image


def open_image_from_url(url, cv2_img_flag=0):
    """Attempt to load an OpenCV image from a URL"""
    # See http://stackoverflow.com/a/13329446/59984
    request = urlopen(url)
    img_array = numpy.asarray(bytearray(request.read()), dtype=numpy.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)
