"""Module for processing attrition data."""

import logging
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd

logger = logging.getLogger(__name__)


class ReadAttritionData(ABC):
    """Abstract class for attrition readers."""

    def __init__(self) -> None:
        """Init of ProcessingAttritionData."""

    @abstractmethod
    def read_from_source(self) -> pd.DataFrame:
        """Read source of information into DF."""

    @abstractmethod
    def process(self) -> pd.DataFrame:
        """Process information and returns pandas DF."""


class WriteAttritionData(ABC):
    # pylint: disable=too-few-public-methods
    """Abstract class for attrition writers."""

    def __init__(self, attrition_data_df: pd.DataFrame) -> None:
        "Init of ProcessingAttritionData"
        self.attrition_data_df = attrition_data_df

    @abstractmethod
    def write(self) -> None:
        """Write processed pandas DF."""


class ReadAttritionDataFromCSV(ReadAttritionData):
    """Read attrition data from a CSV file."""

    def __init__(self, source_csv_str_path: str) -> None:
        "Init of ProcessingAttritionData"
        super().__init__()
        self.source_csv_str_path = source_csv_str_path
        self.attrition_data_df = self.read_from_source()

    def read_from_source(self) -> pd.DataFrame:
        """Read source of information into DF."""
        file_path = Path(self.source_csv_str_path)
        if not file_path.is_file():
            raise RuntimeError(f'The file was not found: {self.source_csv_str_path}')
        return pd.read_csv(file_path)

    def process(self) -> pd.DataFrame:
        """Process information and returns pandas DF."""
        self.attrition_data_df = self._check_employee_counts_equal_to_1()
        self.attrition_data_df = self._remove_column('EmployeeNumber')
        self.attrition_data_df = self._remove_columns_with_single_value()
        return self.attrition_data_df

    def _check_employee_counts_equal_to_1(self) -> pd.DataFrame:
        """Check for a valid EmployeeCount."""
        employee_counts = pd.unique(self.attrition_data_df['EmployeeCount'])
        if len(employee_counts) != 1:
            raise NotImplementedError(f'More than one EmployeeCount found. Only 1 supported. {employee_counts}')

        employee_count = employee_counts.item()
        if employee_count != 1:
            raise NotImplementedError(f'No functionality implemented for EmployeeCount != 1. {employee_count}')

        return self.attrition_data_df.drop(columns=['EmployeeCount'])

    def _remove_column(self, column_to_remove: str) -> pd.DataFrame:
        """Remove EmployeeNumber from data."""
        return self.attrition_data_df.drop(columns=[column_to_remove])

    def _remove_columns_with_single_value(self) -> pd.DataFrame:
        original_columns = list(self.attrition_data_df.columns)
        for column in original_columns:
            column_unique_values = self.attrition_data_df[column].unique()
            if len(column_unique_values) == 1:
                logger.warning(f'Going to remove column: {column}. Contains single value')
                self.attrition_data_df = self._remove_column(column)
        return self.attrition_data_df
