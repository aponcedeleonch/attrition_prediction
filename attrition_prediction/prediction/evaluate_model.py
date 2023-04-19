import logging
from abc import ABC, abstractmethod
from itertools import chain, combinations
from typing import Iterable
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, ShuffleSplit, train_test_split
from sklearn.metrics import recall_score


logger = logging.getLogger(__name__)


def get_all_possible_combinations_of_list(list_to_get_combination):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3). Taken from python docs."""
    s = list(list_to_get_combination)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def select_one_hot_encoded_columns(processed_data_df, wanted_columns):
        one_hot_encoded_columns = list(processed_data_df.columns)
        selected_columns = [
                            one_hot_column
                            for column in wanted_columns
                            for one_hot_column in one_hot_encoded_columns
                            if column in one_hot_column
                        ]
        return processed_data_df[selected_columns]


class ModelEvaluator:

    def __init__(
                    self,
                    processed_data_df: pd.DataFrame,
                    test_size: float,
                    number_k_splits: int,
                    predictions_features: str,
                    scoring_metric: str
                ) -> None:
        self.processed_data_df = processed_data_df.copy()
        self.test_size = test_size
        self.number_k_splits = number_k_splits
        self.predictions_features = predictions_features
        self.scoring_metric = scoring_metric
        self.data_splitter_for_cross_validation = self._get_data_splitter()
        self.x_features, self.y_predict = self._separate_df_data_for_model_fitting()

    def evaluate(self, model_to_evaluate):
        cross_validate_results_dict =  cross_validate(
                                                    estimator=model_to_evaluate,
                                                    X=self.x_features,
                                                    y=self.y_predict,
                                                    cv=self.data_splitter_for_cross_validation,
                                                    scoring=self.scoring_metric,
                                                    return_train_score=True,
                                                    return_estimator=True
                                                )
        train_scores = cross_validate_results_dict['train_score']
        test_scores = cross_validate_results_dict['test_score']
        fitted_estimators = cross_validate_results_dict['estimator']
        best_estimator = fitted_estimators[np.argmax(test_scores)]
        return train_scores.mean(), test_scores.mean(), best_estimator

    def _separate_df_data_for_model_fitting(self):
        feature_to_predict = self.processed_data_df['Attrition'].to_numpy()
        features_used_for_prediction = select_one_hot_encoded_columns(
                                                                        self.processed_data_df,
                                                                        self.predictions_features
                                                                    ).to_numpy()
        return features_used_for_prediction, feature_to_predict

    def _get_data_splitter(self):
        return ShuffleSplit(n_splits=self.number_k_splits, test_size=self.test_size)


class KFold(ABC):
    """Abstract class for KFold."""

    def __init__(self) -> None:
        """Init of KFold."""

    @abstractmethod
    def _separate_test_and_training_validation(self):
        """Separate into test and training/validation."""

    @abstractmethod
    def _get_single_fold(self):
        """Get the score for a single fold."""
    
    @abstractmethod
    def get_all_folds(self):
        """Get the best model based on the results of each fold."""


class KFoldColumns(KFold):

    def __init__(
                    self,
                    processed_data_df: pd.DataFrame,
                    test_size: float,
                    validation_size: float,
                    columns_to_test: Iterable[str],
                    number_splits: int,
                    scoring_metric: str,
                ) -> None:
        "Init of KFoldColumns"
        super().__init__()
        self.processed_data_df = processed_data_df.copy()
        self.test_size = test_size
        self.validation_size = validation_size
        self.number_splits = number_splits
        self.scoring_metric = scoring_metric
        self.train_data, self.test_data = self._separate_test_and_training_validation()
        all_column_combinations = get_all_possible_combinations_of_list(columns_to_test)
        self.column_combinations_to_try = [
                                                combination
                                                for combination in all_column_combinations
                                                if len(combination) > 17
                                                # if len(combination) > 19
                                            ]
        self.all_folds = []

    def _separate_test_and_training_validation(self):
        return train_test_split(self.processed_data_df, test_size=self.test_size)
    
    def _get_single_fold(self, columns, estimator):
        model_evaluator = ModelEvaluator(
                                        processed_data_df=self.train_data,
                                        test_size=self.validation_size,
                                        number_k_splits=self.number_splits,
                                        predictions_features=columns,
                                        scoring_metric=self.scoring_metric
                                    )
        return model_evaluator.evaluate(estimator)

    def get_all_folds(self, estimator):
        self.all_folds = []
        logger.warning(f'Going to try {len(self.column_combinations_to_try)} combinations.')
        for i_comb, columns in enumerate(self.column_combinations_to_try):
            self.all_folds.append(self._get_single_fold(columns, estimator))
            if i_comb % 100 == 0 and i_comb != 0:
                logger.warning(f'Obtained {i_comb} combinations.')
        return self.all_folds
    
    def get_top_n_folds(self, top_n):
        if len(self.all_folds) == 0:
            logger.warning('Cannot get top-N folds. Folds have not been obtained.')
            return
        validation_scores = [validation_score for _, validation_score, _ in self.all_folds]
        idx_sorted_validation_scores = np.argsort(validation_scores)
        idx_top_n_validation_scores = idx_sorted_validation_scores[-top_n:]
        top_n_folds = [
                        (self.all_folds[idx], self.column_combinations_to_try[idx])
                        for idx in idx_top_n_validation_scores
                    ]
        return top_n_folds
    
    def get_score_on_test_set(self, estimator, columns):
        feature_to_predict = self.test_data['Attrition'].to_numpy()
        features_used_for_prediction = select_one_hot_encoded_columns(
                                                                        self.test_data,
                                                                        columns
                                                                    ).to_numpy()
        predictions = estimator.predict(features_used_for_prediction)
        return recall_score(feature_to_predict, predictions)
