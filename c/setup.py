#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# inplace extension module
_intersection_detector = Extension("_intersection_detector",
                   ["intersection_detector.i","intersection_detector.c"],
                   include_dirs = [numpy_include],
                   )

# NumyTypemapTests setup
setup(  name        = "intersection_detector function",
        description = "Test rays for intersection with an obstacle grid map.",
        author      = "J.Zhu",
        version     = "0.0",
        ext_modules = [_intersection_detector]
        )
