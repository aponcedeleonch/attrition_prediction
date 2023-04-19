"""Module for processing attrition data."""

import logging
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
from attrition_prediction.data_processing.type_of_columns import CATEGORIZED_COLUMNS

logger = logging.getLogger(__name__)


class ReadAttritionData(ABC):
    """Abstract class for attrition readers."""

    def __init__(self) -> None:
        """Init of ProcessingAttritionData."""

    @abstractmethod
    def read_from_source(self) -> pd.DataFrame:
        """Read source of information into DF."""

    @abstractmethod
    def process_for_plotting(self) -> pd.DataFrame:
        """Process information and returns pandas DF."""

    @abstractmethod
    def process_for_prediction(self) -> pd.DataFrame:
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

    def _common_processing(self) -> pd.DataFrame:
        """Process information and returns pandas DF."""
        self.attrition_data_df = self._check_employee_counts_equal_to_1()
        self.attrition_data_df = self._remove_column('EmployeeNumber')
        self.attrition_data_df = self._remove_columns_with_single_value()
        self.attrition_data_df = self._convert_all_columns_to_categories()
        return self.attrition_data_df

    def process_for_plotting(self) -> pd.DataFrame:
        return self._common_processing()
    
    def process_for_prediction(self):
        self.attrition_data_df = self._common_processing()
        self.attrition_data_df = self._convert_attrition_column_to_bool()
        return self._convert_feature_columns_to_one_hot_encoded()

    def _convert_attrition_column_to_bool(self):
        self.attrition_data_df.loc[self.attrition_data_df['Attrition'] == 'Yes', 'Attrition'] = 1
        self.attrition_data_df.loc[self.attrition_data_df['Attrition'] == 'No', 'Attrition'] = 0
        self.attrition_data_df['Attrition'] = pd.to_numeric(self.attrition_data_df['Attrition'])
        return self.attrition_data_df
    
    def _convert_feature_columns_to_one_hot_encoded(self):
        all_columns_but_attrition = set(self.attrition_data_df.columns) - set(['Attrition'])
        one_hot_encoded_df = self.attrition_data_df[list(all_columns_but_attrition)].copy().astype(str)
        one_hot_encoded_df = pd.get_dummies(one_hot_encoded_df)
        one_hot_encoded_df['Attrition'] = self.attrition_data_df['Attrition']
        return one_hot_encoded_df
    
    def _convert_all_columns_to_categories(self):
        original_columns = list(self.attrition_data_df.columns)
        for column in original_columns:
            self.attrition_data_df = self._transform_column_to_categories(column)
        return self.attrition_data_df

    def _transform_column_to_categories(self, column):
        if column in CATEGORIZED_COLUMNS:
            return self.attrition_data_df

        logger.warning(f'Transforming {column} to categorized column.')
        new_column_name = f'{column}Categorized'
        self.attrition_data_df = self._get_quantiles_of_column(column, new_column_name)
        self.attrition_data_df[column] = self.attrition_data_df[new_column_name]
        self.attrition_data_df = self._remove_column(new_column_name)
        return self.attrition_data_df
    
    def _get_quantiles_of_column(self, column, new_column_name):
        quantiles_to_get = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantiles_values = self.attrition_data_df[column].quantile(quantiles_to_get)
        self.attrition_data_df[new_column_name] = (
                                                    f'[0,0.1)Q\n'
                                                    f'({self.attrition_data_df[column].min():.1f}, '
                                                    f'{quantiles_values[0.1]:.1f})'
                                                )
        for quantile, next_quantile in zip(quantiles_to_get[:-1], quantiles_to_get[1:]):
            quantile_value = quantiles_values[quantile]
            next_quantile_value = quantiles_values[next_quantile]
            self.attrition_data_df.loc[
                                        self.attrition_data_df[column].between(
                                                                                quantile_value,
                                                                                next_quantile_value,
                                                                                inclusive='left'
                                                                            ),
                                        new_column_name
                                    ] = (
                                            f'[{quantile}, {next_quantile})Q\n'
                                            f'[{quantile_value:.1f}, {next_quantile_value:.1f})'
                                        )
        self.attrition_data_df.loc[
                                    self.attrition_data_df[column] >= quantiles_values[0.9],
                                    new_column_name
                                ] = (
                                        f'[0.9,1]Q\n'
                                        f'[{quantiles_values[0.9]:.1f}, '
                                        f'{self.attrition_data_df[column].max():.1f}]'
                                    )
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
