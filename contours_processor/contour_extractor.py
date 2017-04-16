import re
import os.path
import numpy as np
from glob import glob

import dicom
from dicom.errors import InvalidDicomError

from PIL import Image, ImageDraw


class ContourFileExtractor(object):
    """Generate Dataset matching Contour and Dicoms"""

    def _dicom_filename_extractor(self, dicom_root, contour_folder, contour_filename):

        dicom_folder = self._contour_dicom_folder_map.get(contour_folder)
        dicom_match = re.search(r"IM-0001-(\d{4})-icontour-manual.txt", contour_filename)

        if dicom_folder and dicom_match:
            dicom_filename = str(int(dicom_match.group(1))) + ".dcm"
            return os.path.join(dicom_root, dicom_folder, dicom_filename)

    def _is_valid_contours(self, coords):
        # TODO Additional Validation beyond first and last record
        if coords is None or len(coords) < 2 or coords[0] != coords[-1]:
            return False
        return True

    def __init__(self,
            contour_top_folder,
            dicom_top_folder,
            contour_type="i-contours",
            target_size=(256, 256),
            padding=0,
            on_error="raise",  # raise, skip the file
            contour_dicom_folder_map=None,
            dicom_filename_extractor=None):
        """Create a Dataset with given parameters"""
        self._contour_root = contour_top_folder
        self._dicom_root = dicom_top_folder
        self._contour_type = contour_type
        self._target_size = target_size
        self._contour_dicom_folder_map = contour_dicom_folder_map
        if dicom_filename_extractor is None:
            self.dicom_filename_extractor = self._dicom_filename_extractor
        else:
            self.dicom_filename_extractor = dicom_filename_extractor

    def _parse_dicom_file(self, filename):
        """Parse the given DICOM filename

        :param filename: filepath to the DICOM file to parse
        :return: dictionary with DICOM image data
        """
        try:
            dcm = dicom.read_file(filename)
            dcm_image = dcm.pixel_array
            # TODO: Crop and Pad DCM image to target size
            try:
                intercept = dcm.RescaleIntercept
            except AttributeError:
                intercept = 0.0
            try:
                slope = dcm.RescaleSlope
            except AttributeError:
                slope = 0.0

            if intercept != 0.0 and slope != 0.0:
                dcm_image = dcm_image * slope + intercept
            return dcm_image
        except InvalidDicomError:
            return None

    def _parse_contour_file(self, filename):
        """Parse the given contour filename

        :param filename: filepath to the contourfile to parse
        :return: list of tuples holding x, y coordinates of the contour
        """
        coords_lst = []

        with open(filename, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                coords_lst.append((x_coord, y_coord))

        if self._is_valid_contours(coords_lst):
            img = Image.new(mode='L', size=self._target_size, color=0)
            ImageDraw.Draw(img).polygon(xy=coords_lst, outline=0, fill=1)
            mask = np.array(img).astype(bool)
            return mask

    def _get_contour_files(self, shuffle):
        contour_files = []
        for contour_folder in self._contour_dicom_folder_map.keys():
            contour_folder_files = glob(os.path.join(
                self._contour_root, contour_folder, self._contour_type, "*.txt"))
            contour_files.extend(contour_folder_files)

        if shuffle:
            contour_files = np.random.permutation(contour_files)
        return contour_files

    def _extract_dicom_contour_file(self, contour_path):
        contour_folder, contour_type, contour_filename = self._split_contour_path(contour_path)
        dicom_path = self.dicom_filename_extractor(self._dicom_root, contour_folder, contour_filename)
        if dicom_path is not None:
            dicom_data = self._parse_dicom_file(dicom_path)
        contour_data = self._parse_contour_file(contour_path)

        return dicom_data, contour_data

    def _split_contour_path(self, contour_path):
        # example: SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt
        contour_file_tree = contour_path.split("/")
        contour_folder, contour_type, contour_filename = contour_file_tree[-3:]
        return contour_folder, contour_type, contour_filename

    def _get_dicom_contour_outfile(self, contour_path):
        contour_folder, contour_type, contour_filename = self._split_contour_path(contour_path)
        output_filename = contour_filename.split(".")[0]
        output_filename = "{}-{}.npy".format(contour_folder, output_filename)

        return output_filename

    def create_contour_dicoms_generator(self, shuffle=False):
        """Return batch of Contours and Dicoms"""
        contour_files = self._get_contour_files(shuffle)
        for contour_path in contour_files:
            dicom_data, contour_data = self._extract_dicom_contour_file(contour_path)
            if dicom_data is not None and contour_data is not None:
                yield dicom_data, contour_data
            else:
                # TODO LOG Error in reading / matching file
                pass

    def create_contour_dicoms(self, output_dir, n_samples=None, shuffle=False):
        """Return batch of Contours and Dicoms"""
        contour_files = self._get_contour_files(shuffle)

        for idx, contour_path in enumerate(contour_files):
            if n_samples and idx > n_samples:
                break
            dicom_data, contour_data = self._extract_dicom_contour_file(contour_path)
            if dicom_data is not None and contour_data is not None:
                dicom_contour_data = np.stack([dicom_data, contour_data])
                output_filename = self._get_dicom_contour_outfile(contour_path)
                np.save(os.path.join(output_dir, output_filename), dicom_contour_data)
            else:
                # TODO LOG Error in reading / matching file
                pass
