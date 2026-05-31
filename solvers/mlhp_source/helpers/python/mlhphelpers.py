# NeuralMech extension to mlhp. License: See mlhp/LICENSE
#
# Thin python wrapper exposing the custom helper integrands. Mirrors mlhp.py:
# the compiled pymlhphelpers module sits next to this file in the build's bin
# directory, so add that directory to the path before importing from it.

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pymlhphelpers import *
