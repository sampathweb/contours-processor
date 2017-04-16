from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import numpy.testing as npt
from contours_processor import ContourFileExtractor, ContourDataset


def test_create():
    """
    Testing creation of Extractor and Dataset"""

    contour_top_folder = "data/contourfiles/"
    dicom_top_folder = "data/dicoms/"
    contour_extractor = ContourFileExtractor(contour_top_folder, dicom_top_folder)
    contours_dset = ContourDataset(contour_dicom_folder="data/contour_dicom_processed/")
