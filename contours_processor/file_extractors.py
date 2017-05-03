import re
import os.path
import numpy as np
from glob import glob
import logging

import dicom
from dicom.errors import InvalidDicomError

from PIL import Image, ImageDraw

from . import logger
from .utils import dump_dataset
from .exceptions import InvalidDatasetError


class ContourFileExtractor(object):
    """Parse Contour and Dicoms Files for use with ContourDataset.

    Files can be processed into Hdf5 formats or can be used as on-line Generators.
    """

    def _contour_secondary_filepath_extractor(self, secondary_type, contour_filepath):
        """Return the filepath for the secondary contour file"""
        contour_type_mapper = {
            "i-contours": "icontour",
            "o-contours": "ocontour"
        }
        *location, folder, contour_type, filename = self._split_contour_path(contour_filepath)
        if contour_type in contour_type_mapper and secondary_type in contour_type_mapper:
            filename = filename.replace(
                contour_type_mapper[contour_type],
                contour_type_mapper[secondary_type])

        filepath = os.path.join(*location, folder, secondary_type, filename)
        if os.path.isfile(filepath):
            return filepath

    def _dataset_validator(self, datasets):
        """Return True if dataset is valid.

        Verify that i-contour is within the boundaries of o-contours when they exist
        Additional validations as necessary.
        """
        if not datasets:
            return False  # Empty Dataset

        # Verify that All the datasets have same Shape
        dataset_items = list(datasets.items())
        dataset_shape = dataset_items[0][1].shape
        for key, data_array in dataset_items[1:]:
            if data_array.shape != dataset_shape:
                return False  # All arrays need to be of same shape

        # Verify that i-contours is a subset of o-contors
        if "i-contours" in datasets and "o-contours" in datasets:
            invalid_pixels = np.where((datasets["o-contours"] == 0) &
                (datasets["i-contours"] > 0))[0]
            if len(invalid_pixels) > 0:
                return False  # i-contours should be fully contained in o-contours
        return True

    def _dicom_filepath_extractor(self,
            dicom_root,
            contour_filepath,
            contour_dicom_folder_map=None):
        """Return the dicom filepath for the given contour filename"""
        contour_file_tree = contour_filepath.split("/")
        contour_folder, contour_type, contour_filename = contour_file_tree[-3:]
        dicom_folder = contour_dicom_folder_map.get(contour_folder)

        dicom_match = re.search(r"IM-0001-(\d{4})-[io]contour.*?\.txt", contour_filename)

        dicom_filename = None
        if dicom_match:
            dicom_filename = str(int(dicom_match.group(1))) + ".dcm"
        if dicom_folder and dicom_filename:
            return os.path.join(dicom_root, dicom_folder, dicom_filename)

    def _is_valid_contours(self, coords):
        # TODO Additional Validation
        if coords is None or len(coords) < 2:
            return False
        return True

    def _log_error(self, err_msg):
        """Raise Exception if action is set otherwise log as warnings"""
        if self._on_error_action == "raise":
            raise InvalidDatasetError(err_msg)
        else:
            logger.warning(err_msg)

    def __init__(self,
            contour_top_folder,
            dicom_top_folder,
            primary_contour="o-contours",
            secondary_contours=None,
            target_size=(256, 256),
            padding=0,
            on_error_action="raise",  # raise, skip the file
            contour_dicom_folder_map=None,
            dicom_filepath_extractor=None,
            contour_secondary_filepath_extractor=None,
            dataset_validator=None,
            save_filepath_extractor=None):
        """Create a Dataset with given parameters"""
        self._contour_root = contour_top_folder
        self._dicom_root = dicom_top_folder
        self.primary_contour = primary_contour
        self.secondary_contours = secondary_contours
        if self.secondary_contours is None:
            self.secondary_contours = []
        self._target_size = target_size
        self._padding = padding
        self._on_error_action = on_error_action
        self._contour_dicom_folder_map = contour_dicom_folder_map
        if dicom_filepath_extractor is None:
            self.dicom_filepath_extractor = self._dicom_filepath_extractor
        else:
            self.dicom_filepath_extractor = dicom_filepath_extractor

        if contour_secondary_filepath_extractor:
            self.contour_secondary_filepath_extractor = contour_secondary_filepath_extractor
        else:
            self.contour_secondary_filepath_extractor = self._contour_secondary_filepath_extractor
        if dataset_validator:
            self.dataset_validator = dataset_validator
        else:
            self.dataset_validator = self._dataset_validator
        if save_filepath_extractor:
            self.save_filepath_extractor = save_filepath_extractor
        else:
            self.save_filepath_extractor = self._save_filepath_extractor

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
        if filename is None:
            return None

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
                self._contour_root, contour_folder, self.primary_contour, "*.txt"))
            contour_files.extend(contour_folder_files)

        if shuffle:
            contour_files = np.random.permutation(contour_files)
        return contour_files

    def _extract_dicom_contour_file(self, contour_path):
        """Return Prased Datasets and Sources.

        Navigate the Contour Path to get Dicom and Contour Files.
        Extract them as numpy arrays in Datasets.
        """
        datasets = {}
        sources = {}
        # Parse Dicoms
        dicom_path = self.dicom_filepath_extractor(self._dicom_root, contour_path,
            self._contour_dicom_folder_map)
        dicom_data = self._parse_dicom_file(dicom_path)
        if dicom_path and dicom_data is not None:
            datasets["dicom"] = dicom_data
            sources["dicom"] = dicom_path
        else:
            err_msg = "Dicom File Parse Error: {}".format(dicom_path)
            self._log_error(err_msg)

        # Parse Primary Contour
        contour_data = self._parse_contour_file(contour_path)
        if contour_data is not None:
            datasets[self.primary_contour] = contour_data
            sources[self.primary_contour] = contour_path
        else:
            err_msg = "Dicom File Parse Error: {}".format(dicom_path)
            self._log_error(err_msg)

        # Parse Secondary Contours
        for secondary_contour in self.secondary_contours:
            contour_seconday_filepath = self.contour_secondary_filepath_extractor(
                secondary_contour, contour_path)
            contour_data = self._parse_contour_file(contour_seconday_filepath)
            if contour_data is not None:
                datasets[secondary_contour] = contour_data
                sources[secondary_contour] = contour_seconday_filepath

        return datasets, sources

    def _split_contour_path(self, contour_path):
        # example: SC-HF-I-1/i-contours/IM-0001-0048-icontour-manual.txt
        contour_file_tree = contour_path.split("/")
        return contour_file_tree

    def _save_filepath_extractor(self, output_dir, contour_path, sources):
        """Return output filepath to save processed dataset"""
        *_, contour_folder, contour_type, contour_filename = self._split_contour_path(contour_path)
        dicom_match = re.search(r"(IM-\d{4}-\d{4})-.*\.txt", contour_filename)
        if dicom_match:
            file_prefix = dicom_match.group(1)
            output_filename = "{}-{}-dicom-contours.h5".format(contour_folder, file_prefix)
            return os.path.join(output_dir, output_filename)

    def datasets_generator(self, shuffle=False):
        """Return batch of Datasets and Sources from Dicom / Contours file"""
        contour_files = self._get_contour_files(shuffle)
        for contour_path in contour_files:
            datasets, sources = self._extract_dicom_contour_file(contour_path)
            if datasets and self.dataset_validator(datasets):
                yield datasets, sources
            else:
                self._log_error("Dataset failed validation {}".format(contour_path))

    def save_datasets(self, output_dir, n_samples=None, shuffle=False):
        """Save Datasets and Sources metadata in output directory"""
        contour_files = self._get_contour_files(shuffle)

        for idx, contour_path in enumerate(contour_files):
            if n_samples and idx > n_samples:
                break
            datasets, sources = self._extract_dicom_contour_file(contour_path)
            output_filepath = self.save_filepath_extractor(output_dir, contour_path, sources)
            if output_filepath and self.dataset_validator(datasets):
                dump_dataset(output_filepath, datasets, sources)
            else:
                self._log_error("Dataset failed validation {}".format(output_filepath))
