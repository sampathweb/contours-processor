import os.path
from pickle import UnpicklingError
from glob import glob
import numpy as np

from . import logger
from .utils import load_dataset
from .exceptions import InvalidDatasetError


class ContourDataset(object):
    """Generate Dataset for Dicoms and Contour Pair"""

    def __init__(self,
            x_channels,
            y_channels=None,
            include_sources=False,
            contour_dicom_generator=None,
            contour_dicom_folder=None,
            on_error_action="raise",
            target_size=(224, 224),
            padding=0,
            crop_mode="center"):  # center or random cropping
        """Create a Dataset with given parameters"""
        if contour_dicom_folder and contour_dicom_generator:
            raise InvalidDatasetError("Please specify only the folder for files or the generator object")
        if contour_dicom_folder is None and contour_dicom_generator is None:
            raise InvalidDatasetError("Both the folder for files and the generator object can't be None")

        self.x_channels = x_channels
        self.y_channels = y_channels
        self._include_sources = include_sources
        self._contour_dicom_folder = contour_dicom_folder
        self._contour_dicom_generator = contour_dicom_generator
        self._on_error_action = on_error_action
        # TODO: Resize, Padding and Cropping in generated Batches
        self._target_size = target_size
        self._padding = padding
        self._crop_mode = "center"

    def _log_error(self, err_msg):
        """Raise Exception if action is set otherwise log as warnings"""
        if self._on_error_action == "raise":
            raise InvalidDatasetError(err_msg)
        else:
            logger.warning(err_msg)

    def _contour_folder_gen(self, contour_files):

        for contour_file in contour_files:
            try:
                yield load_dataset(contour_file)
            except (FileNotFoundError, UnpicklingError) as e:
                raise InvalidDatasetError("{} - File IO Error".format(contour_file))

    def _parse_channels(self, dataset, channels):
        """Extract channels from dataset.  Raise error if channel data not found

        :param dataset: Dictionary of datasets
        :param channels: Single channel (passed as string) or List of Channels
        """
        if type(channels) == str:
            data = dataset.get(channels)
            if data is None:
                raise ValueError
        else:
            try:
                data = [dataset.get(channel) for channel in channels]
            except Exception:
                raise ValueError
            for channel_data in data:
                if channel_data is None:
                    raise ValueError
        return data

    def generate_batch(self, batch_size=8, shuffle=True):
        """Return batch of Dicoms and Contours

        :param batch_size: Length of each batch.  Default to 8
        :param shuffle: When True, randomly shuffled batch data
        :return: Numpy Array with shape (batch_size, width, height)
        """
        if self._contour_dicom_folder:
            contour_files = glob(os.path.join(self._contour_dicom_folder, "*.h5"))
            if shuffle:
                contour_files = np.random.permutation(contour_files)
            contours_generator = self._contour_folder_gen(contour_files)
        else:
            contours_generator = self._contour_dicom_generator

        x_batch, y_batch, sources_batch = [], [], []
        batch_idx = 0
        for idx, (dataset, sources) in enumerate(contours_generator):
            if batch_idx > 0 and batch_idx % batch_size == 0:
                if self._include_sources:
                    yield sources_batch, np.array(x_batch), np.array(y_batch)
                else:
                    yield np.array(x_batch), np.array(y_batch)
                x_batch, y_batch, sources_batch = [], [], []
                batch_idx = 0
            try:
                x_data = self._parse_channels(dataset, self.x_channels)
                y_data = self._parse_channels(dataset, self.y_channels)
                x_batch.append(x_data)
                y_batch.append(y_data)
                sources_batch.append(sources)
                batch_idx += 1
            except ValueError:
                # Log Error
                err_msg = "Missing all channels in {}".format(sources["filename"])
                self._log_error(err_msg)

        if self._include_sources:
            yield sources_batch, np.array(x_batch), np.array(y_batch)
        else:
            yield np.array(x_batch), np.array(y_batch)
