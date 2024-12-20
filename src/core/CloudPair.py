import os
import re


class CloudPair:
    """
    Represents a pairing of dense and precision point clouds, along with associated metadata such as
    log files, date, and index. This class enforces consistency between these elements based on their timestamps.
    """

    def __init__(self):
        """
        Initializes a CloudPair instance with default values for its attributes.
        """
        self._dense: str = None  # Path to the dense point cloud file.
        self._prec: str = None  # Path to the precision point cloud file.
        self._log: str = None  # Path to the log file associated with the point cloud.
        self._date: str = None  # Timestamp extracted from the dense file's name.
        self._index: str = None  # Index extracted from the dense file's name.

        self.m3c2_param = None  # Placeholder for M3C2-specific parameters.

    @property
    def dense(self):
        """
        Getter for the dense point cloud file path.

        Returns:
            str: Path to the dense point cloud file.
        """
        return self._dense

    @property
    def prec(self):
        """
        Getter for the precision point cloud file path.

        Returns:
            str: Path to the precision point cloud file.
        """
        return self._prec

    @property
    def log(self):
        """
        Getter for the log file path.

        Returns:
            str: Path to the log file.
        """
        return self._log

    @property
    def date(self):
        """
        Getter for the timestamp extracted from the dense point cloud file name.

        Returns:
            str: Timestamp as a string.
        """
        return self._date

    @property
    def index(self):
        """
        Getter for the index extracted from the dense point cloud file name.

        Returns:
            str: Index as a string.
        """
        return self._index

    @dense.setter
    def dense(self, dense_path: str):
        """
        Setter for the dense point cloud file path. Extracts and sets the timestamp and index
        from the file name if it matches the expected pattern.

        Args:
            dense_path (str): Path to the dense point cloud file.

        Raises:
            ValueError: If the file name does not match the expected pattern.
        """
        pattern = r'\b\d{1,4}_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}.'

        if re.search(pattern, os.path.basename(dense_path)):
            self._dense = dense_path
            self._date = os.path.splitext(os.path.basename(dense_path).split('_')[1])[0]
            self._index = os.path.splitext(os.path.basename(dense_path).split('_')[0])
        else:
            raise ValueError("Dense file name does not match the expected pattern.")

    @prec.setter
    def prec(self, prec_path: str):
        """
        Setter for the precision point cloud file path. Ensures the timestamp matches the dense file's timestamp.

        Args:
            prec_path (str): Path to the precision point cloud file.

        Raises:
            Exception: If the dense file path has not been set.
            ValueError: If the precision file's timestamp does not match the dense file's timestamp.
        """
        if self._date in prec_path and self.dense is not None:
            self._prec = prec_path
        elif self.dense is None:
            raise Exception('Dense cloud has not been set. Abort.')
        else:
            raise ValueError('Precision timestamp does not match the dense timestamp.')

    @log.setter
    def log(self, log_path: str):
        """
        Setter for the log file path. Ensures the timestamp matches the dense file's timestamp.

        Args:
            log_path (str): Path to the log file.

        Raises:
            Exception: If the dense file path has not been set.
            ValueError: If the log file's timestamp does not match the dense file's timestamp.
        """
        if self._date in log_path and self.dense is not None:
            self._log = log_path
        elif self.dense is None:
            raise Exception('Dense parameter has not been set. Abort.')
        else:
            raise ValueError('Log timestamp does not match the dense timestamp.')
