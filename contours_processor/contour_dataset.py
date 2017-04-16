import os.path
from pickle import UnpicklingError
from glob import glob
import numpy as np


class InvalidDataset(Exception):
    """Exception Raised on invalid dataset"""

    def __init__(self, message):
        """Raise exception with message"""
        self.message = message
        super(Exception, self).__init__(message)


class ContourDataset(object):
    """Generate Dataset for Dicoms and Contour Pair"""

    def __init__(self,
            contour_dicom_generator=None,
            contour_dicom_folder=None,
            target_size=(224, 224),
            padding=0,
            crop_mode="center"):  # center or random cropping
        """Create a Dataset with given parameters"""
        if contour_dicom_folder and contour_dicom_generator:
            raise InvalidDataset("Please specify only the folder for files or the generator object")
        if contour_dicom_folder is None and contour_dicom_generator is None:
            raise InvalidDataset("Both the folder for files and the generator object can't be None")

        self._contour_dicom_folder = contour_dicom_folder
        self._contour_dicom_generator = contour_dicom_generator

        # TODO: Resize, Padding and Cropping in generated Batches
        self._target_size = target_size
        self._padding = padding
        self._crop_mode = "center"

    def _contour_folder_gen(self, contour_files):

        for contour_file in contour_files:
            try:
                yield np.load(contour_file)
            except (FileNotFoundError, UnpicklingError) as e:
                raise InvalidDataset("{} - File IO Error".format(contour_file))

    def generate_batch(self, batch_size=8, shuffle=True):
        """Return batch of Dicoms and Contours

        :param batch_size: Length of each batch.  Default to 8
        :param shuffle: When True, randomly shuffled batch data
        :return: Numpy Array with shape (batch_size, width, height)
        """
        if self._contour_dicom_folder:
            contour_files = glob(os.path.join(self._contour_dicom_folder, "*.npy"))
            if shuffle:
                contour_files = np.random.permutation(contour_files)
            contours_generator = self._contour_folder_gen(contour_files)
        else:
            contours_generator = self._contour_dicom_generator

        dicom_batch, contour_batch = [], []
        for idx, (dicom_data, contour_data) in enumerate(contours_generator):
            if idx > 0 and idx % batch_size == 0:
                yield np.array(dicom_batch), np.array(contour_batch)
                dicom_batch, contour_batch = [], []
            dicom_batch.append(dicom_data)
            contour_batch.append(contour_data)
        yield np.array(dicom_batch), np.array(contour_batch)
