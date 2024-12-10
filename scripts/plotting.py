#!/usr/bin/env python3
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np


class Plotting:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, bins=30):
        """
        plots histogram for all columns in a Dataframe.
        Args:
            data:DataFrame  the dataset to plot histogram
            bins:int: Number of bins for the histogram
            figsize: tuple: Figure size for the plot
            title:str: title for the overall plot
        """
        self.data.hist(bins=bins, figsize=(20, 15))
        plt.suptitle("Histograms of all variables in the dataset", fontsize=16)
        plt.show()

    def plot_heatmap(self, data: DataFrame):
        """
        plots a heatmap of the correlation matrix
        Args:
            data:DataFrame  the dataset to plot heatmap
        """
        numerical_features = self.data.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12, 10))
        sns.heatmap(data[numerical_features].corr(), annot=True, cmap="coolwarm")
        plt.title("Heatmap of the correlation matrix")
        plt.show()
