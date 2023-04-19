from typing import Iterable, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class ResultsPlotting:
    """Class to plot attrition data."""

    def __init__(self, base_fontsize: int) -> None:
        """Init of AttritionPlotting."""
        self.base_fontsize = base_fontsize

    def plot_train_scores(self, train_scores, models):
        x_positions = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')
        ax.scatter(x_positions, train_scores, label='Training', s=self.base_fontsize ** 2)
        ax.set_xticks(x_positions, models, fontsize=self.base_fontsize)
        ax.legend(loc='upper left', fontsize=self.base_fontsize + 2)
        ax.set_xlabel('Models', fontsize=self.base_fontsize + 2)
        ax.set_ylabel('Recall', fontsize=self.base_fontsize + 2)
        ax.tick_params(axis='both', which='major', labelsize=self.base_fontsize)
        ax.set_title('Recall results by Estimator', fontsize=self.base_fontsize + 4)
        return fig, ax, x_positions
    
    def plot_train_and_validation_scores(self, train_scores, validation_scores, label_validation, models):
        fig, ax, x_positions = self.plot_train_scores(train_scores, models)
        ax.scatter(x_positions, validation_scores, label=label_validation, s=self.base_fontsize ** 2)
        ax.legend(loc='upper left', fontsize=self.base_fontsize + 2)
        return fig, ax, x_positions
    
    def plot_train_validation_and_test_scores(self, train_scores, validation_scores, test_scores, models):
        fig, ax, x_positions = self.plot_train_and_validation_scores(
                                                                        train_scores,
                                                                        validation_scores,
                                                                        'Validation',
                                                                        models
                                                                    )
        ax.scatter(x_positions, test_scores, label='Testing', s=self.base_fontsize ** 2)
        ax.legend(loc='upper left', fontsize=self.base_fontsize + 2)
        return fig, ax
