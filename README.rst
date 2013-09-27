Experimental image mining using OpenCV
======================================

Attempting to build tools to mine interesting data from large collections of scanned images

Current
-------

* bin/locate-thumbnail: `find the location of a thumbnail in the source image <http://chris.improbable.org/2013/06/30/reconstructing-thumbnails-using-opencv/>`_
* bin/extract-figures: `locate interesting non-text elements (images, figures, tables, etc.) on scanned book pages <http://chris.improbable.org/2013/08/31/extracting-images-from-scanned-pages/>`_

Prerequisites
-------------

* Python 2.6+
* OpenCV
* numpy

Using Mac Homebrew this should install cleanly::

    brew install python numpy opencv
