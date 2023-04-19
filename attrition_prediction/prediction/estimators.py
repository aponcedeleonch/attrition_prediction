import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RandomEstimator(BaseEstimator):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.proportion_of_trues = np.mean(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Check if fit has been called
        # check_is_fitted(self)
        X = check_array(X)

        random_val_for_predictions = np.random.random(X.shape[0])
        random_predictions = np.ones_like(random_val_for_predictions)
        random_predictions[random_val_for_predictions > self.proportion_of_trues] = 0
        return random_predictions
