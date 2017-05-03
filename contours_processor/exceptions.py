from . import logger


class InvalidDatasetError(Exception):
    """Exception Raised on invalid dataset"""

    def __init__(self, message):
        """Raise exception with message"""
        self.message = message
        logger.error(message)
        super(Exception, self).__init__(message)
