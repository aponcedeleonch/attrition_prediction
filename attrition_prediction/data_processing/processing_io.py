from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd

class ReadAttritionData(ABC):

    def __init__(self):
        """Init of ProcessingAttritionData."""
        self.attrition_data_df = None

    @abstractmethod
    def _read_from_source(self):
        """Read source of information into DF."""
        pass

    @abstractmethod
    def process(self):
        """Process information and returns pandas DF."""
        pass


class WriteAttritionData(ABC):

    def __init__(self, attrition_data_df):
        "Init of ProcessingAttritionData"
        self.attrition_data_df = attrition_data_df

    @abstractmethod
    def write(self):
        """Write processed pandas DF."""

class ReadAttritionDataFromCSV(ReadAttritionData):

    def __init__(self, source_csv_str_path):
        "Init of ProcessingAttritionData"
        super().__init__()
        self.source_csv_str_path = source_csv_str_path
        self.attrition_data_df = self._read_from_source()

    def _read_from_source(self):
        """Read source of information into DF."""
        file_path = Path(self.source_csv_str_path)
        if not file_path.is_file():
            raise RuntimeError(f'The file was not found: {self.source_csv_str_path}')
        return pd.read_csv(file_path)

    def process(self):
        """Process information and returns pandas DF."""
        self.attrition_data_df = self._check_employee_counts_equal_to_1()
        self.attrition_data_df = self._remove_employee_number()
        return self.attrition_data_df
    
    def _check_employee_counts_equal_to_1(self):
        """Check for a valid EmployeeCount."""
        employee_counts = pd.unique(self.attrition_data_df['EmployeeCount'])
        if len(employee_counts) != 1:
            raise NotImplementedError(f'More than one EmployeeCount found. Only 1 supported. {employee_counts}')
        else:
            employee_count = employee_counts.item()
            if employee_count != 1:
                raise NotImplementedError(f'No functionality implemented for EmployeeCount != 1. {employee_count}')
        return self.attrition_data_df.drop(columns=['EmployeeCount'])
    
    def _remove_employee_number(self):
        """Remove EmployeeNumber from data."""
        return self.attrition_data_df.drop(columns=['EmployeeNumber'])
