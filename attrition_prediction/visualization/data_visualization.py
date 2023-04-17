from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AttritionPlotting:

    def __init__(self, base_column: str, attrition_data_df: pd.DataFrame, base_fontsize: int) -> None:
        self.base_column = base_column
        self.attrition_data_df = attrition_data_df.copy()
        self.base_column_values = list(pd.unique(self._get_column_from_df(self.base_column)))
        self.base_fontsize = base_fontsize
        self.CATEGORIZED_COLUMNS = [
                                    # 'Age',
                                    'BusinessTravel',
                                    # 'DailyRate',
                                    'Department',
                                    # 'DistanceFromHome',
                                    'Education',
                                    'EducationField',
                                    'EnvironmentSatisfaction',
                                    'Gender',
                                    # 'HourlyRate',
                                    'JobInvolvement',
                                    'JobLevel',
                                    'JobRole',
                                    'JobSatisfaction',
                                    'MaritalStatus',
                                    # 'MonthlyIncome',
                                    # 'MonthlyRate',
                                    # 'NumCompaniesWorked',
                                    'Over18',
                                    'OverTime',
                                    # 'PercentSalaryHike',
                                    'PerformanceRating',
                                    # 'RelationshipSatisfaction',
                                    # 'StandardHours',
                                    'StockOptionLevel',
                                    # 'TotalWorkingYears',
                                    'TrainingTimesLastYear',
                                    'WorkLifeBalance',
                                    # 'YearsAtCompany',
                                    # 'YearsInCurrentRole',
                                    # 'YearsSinceLastPromotion',
                                    # 'YearsWithCurrManager'
                                ]

    def _get_column_from_df(self, column: str) -> pd.Series:
        if column not in self.attrition_data_df.columns:
            raise RuntimeError(f'Trying to get column not existent in DF: {column}. Columns: {self.attrition_data_df.columns}')
        return self.attrition_data_df[column]
    
    def _get_columns_from_df(self, columns: Iterable[str]) -> pd.DataFrame:
        are_columns_in_df = [column in self.attrition_data_df.columns for column in columns]
        if not any(are_columns_in_df):
            raise RuntimeError(f'Trying to get column not existent in DF: {columns}. Columns: {self.attrition_data_df.columns}')
        return self.attrition_data_df[columns]

    def _plot_barplot(self, column_to_plot: str, figure_title: str, plot_relative_values: bool, rotate_xticks: bool=False) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')

        categories = list(pd.unique(self._get_column_from_df(column_to_plot)))
        categories.sort()
        values_to_plot_df = self._get_columns_from_df([self.base_column, column_to_plot])
        x = np.arange(len(categories))
        width = 0.25
        multiplier = 0
        for column_value in self.base_column_values:
            offset = width * multiplier
            data_from_status = values_to_plot_df.loc[values_to_plot_df[self.base_column] == column_value]
            values_from_categories = data_from_status.groupby(column_to_plot).size()
            values_from_categories = values_from_categories[categories].to_numpy()
            if plot_relative_values:
                values_from_categories = (values_from_categories / values_from_categories.sum()) * 100
                values_from_categories = values_from_categories.round(2)
            rects = ax.bar(x + offset, values_from_categories, width, label=column_value)
            ax.bar_label(rects, padding=4, fontsize=self.base_fontsize - 2)
            multiplier += 1

        if plot_relative_values:
            ylabel = f'Percentage of employees by {self.base_column} value.'
        else:
            ylabel = 'Number of employees.'

        ax.set_ylabel(ylabel, fontsize=self.base_fontsize)
        ax.set_title(figure_title, fontsize=self.base_fontsize + 2)
        if rotate_xticks:
            ax.set_xticks(x + width, categories, rotation=55, ha='right', fontsize=self.base_fontsize)
        else:
            ax.set_xticks(x + width, categories, fontsize=self.base_fontsize)
        ax.legend(loc='upper left', fontsize=self.base_fontsize + 2)

        return fig
    
    def _get_quantiles_of_column(self, column, new_column_name):
        quantiles_to_get = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantiles_values = self.attrition_data_df[column].quantile(quantiles_to_get)
        self.attrition_data_df[new_column_name] = f'[0,0.1)Q\n({self.attrition_data_df[column].min():.1f}, {quantiles_values[0.1]:.1f})'
        for quantile, next_quantile in zip(quantiles_to_get[:-1], quantiles_to_get[1:]):
            quantile_value = quantiles_values[quantile]
            next_quantile_value = quantiles_values[next_quantile]
            self.attrition_data_df.loc[
                                        self.attrition_data_df[column].between(quantile_value, next_quantile_value, inclusive='left'),
                                        new_column_name
                                    ] = f'[{quantile}, {next_quantile})Q\n[{quantile_value:.1f}, {next_quantile_value:.1f})'
        self.attrition_data_df.loc[
                                    self.attrition_data_df[column] >= quantiles_values[0.9],
                                    new_column_name
                                ] = f'[0.9,1]Q\n[{quantiles_values[0.9]:.1f}, {self.attrition_data_df[column].max():.1f}]'
        return self.attrition_data_df

    def plot_column(self, column: str, plot_relative_values: bool=False):
        figure_title = f'Attrition of employees by {column}.'
        rotate_xticks = True if column in ('JobRole', 'MonthlyIncome') else False
        column_to_plot = column
        if column not in self.CATEGORIZED_COLUMNS:
            new_column_name = f'{column}Categorized'
            self.attrition_data_df = self._get_quantiles_of_column(column=column, new_column_name=new_column_name)
            column_to_plot = new_column_name

        return self._plot_barplot(
                                    column_to_plot=column_to_plot,
                                    figure_title=figure_title,
                                    plot_relative_values=plot_relative_values,
                                    rotate_xticks=rotate_xticks
                                )

    def plot_all_columns(self, plot_relative_values: bool=False):
        figs = []
        columns_to_plot = set(self.attrition_data_df.columns) - set(['Attrition'])
        for column in list(columns_to_plot):
            figs.append(self.plot_column(column, plot_relative_values))
        return figs
