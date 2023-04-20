from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt

MARKERS = ['o', 'P', '*']

class ResultsPlotting:
    """Class to plot attrition data."""

    def __init__(self, base_fontsize: int) -> None:
        """Init of ResultsPlotting."""
        self.base_fontsize = base_fontsize

    def plot_scores(self, scores: Iterable, labels: Iterable, models: Iterable):
        """Plot with markers the scores of models."""
        x_positions = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')
        for i_scatter, (score, label) in enumerate(zip(scores, labels)):
            ax.scatter(x_positions, score, label=label, s=(self.base_fontsize + 4) ** 2, marker=MARKERS[i_scatter])
        ax.set_xticks(x_positions, models, fontsize=self.base_fontsize)
        ax.legend(loc='upper left', fontsize=self.base_fontsize + 2)
        ax.set_xlabel('Models', fontsize=self.base_fontsize + 2)
        ax.set_ylabel('Recall', fontsize=self.base_fontsize + 2)
        ax.tick_params(axis='both', which='major', labelsize=self.base_fontsize)
        ax.set_title('Recall results by Estimator', fontsize=self.base_fontsize + 4)
        return fig, ax
