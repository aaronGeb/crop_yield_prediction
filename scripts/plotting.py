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

    def plot_boxplot(self, column: str):
        """
        plots boxplot for all columns in a Dataframe.
        Args:
            data:DataFrame  the dataset to plot boxplot
            figsize: tuple: Figure size for the plot
            title:str: title for the overall plot
        """
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(data=self.data, x=column, orient="h")
        ax.set_title(f"Boxplot of {column}")
        ax.set_xlabel(column)
        ax.ticklabel_format(style="plain", axis="x")
        plt.show()

    def plot_barchart(self, x: str, order="descending"):
        """
        Plots a sorted bar chart for a specified column in the DataFrame.

        Args:
            x (str): The column name to plot on the x-axis.
            order (str): Sorting order, either 'ascending' or 'descending'. Default is 'descending'.
        """
        plt.figure(figsize=(12, 6))

        # Count the occurrences of each category
        value_counts = self.data[x].value_counts()

        # Sort the counts based on the specified order
        if order == "ascending":
            sorted_categories = value_counts.sort_values(ascending=True).index
        else:
            sorted_categories = value_counts.sort_values(ascending=False).index

        # Plot the bar chart with the sorted order
        ax = sns.countplot(data=self.data, x=x, order=sorted_categories)

        # Set plot title and labels
        ax.set_title(f"Barplot of {x} (Sorted {order.capitalize()})")
        ax.set_xlabel(x)
        ax.set_ylabel("Count")

        # Show the plot
        plt.show()

    def plot_pairplot(self):
        """
        plots pairplot for all columns in a Dataframe.
        Args:
            data:DataFrame  the dataset to plot pairplot
            figsize: tuple: Figure size for the plot
            title:str: title for the overall plot
        """
        plt.figure(figsize=(12, 6))
        pair_plot = sns.pairplot(self.data)
        plt.suptitle(
            "Pairplot of all variables in the dataset",
            color="darkred",
            fontsize=20,
            fontweight="bold",
        )

        for ax in pair_plot.axes:
            for row in ax:
                if row:
                    # Get the current title and set it with a color
                    title = row.get_title()
                    row.set_title(
                        title, color="darkred", fontsize=12, fontweight="bold"
                    )

        plt.show()

    def plot_scatter(self, x: str, y: str):
        """plots scatter plot for two columns in a Dataframe
        Args:
            data:DataFrame  the dataset to plot scatter
            x:str: x-axis column name
            y:str: y-axis column name
            figsize: tuple: Figure size for the plot
            title:str: title for the overall plot
        """
        plt.figure(figsize=(12, 6))
        ax = sns.regplot(
            data=self.data,
            x=x,
            y=y,
            scatter_kws={"color": "purple"},
            line_kws={"color": "red"},
        )
        ax.set_title(f"Scatter plot of {x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.show()

    def plot_stackedbar(self, x: str, hue: str, figsize=(12, 6), title=None):
        """Plots a stacked bar plot for crop_type and soil_type.

        Args:
            x (str): x-axis column name ('crop_type').
            hue (str): hue column name ('soil_type').
            figsize (tuple): Figure size for the plot.
            title (str): Title for the overall plot.
        """
        # Create a cross-tabulation of crop_type and soil_type
        pivot_df = self.data.groupby([x, hue]).size().unstack(fill_value=0)

        # Plot the stacked bar chart
        pivot_df.plot(kind="bar", stacked=True, figsize=figsize)

        # Add labels and title
        plt.xlabel(x)
        plt.ylabel("Count")
        plt.title(title if title else f"Stacked Bar Plot of {x} by {hue}")
        plt.legend(title=hue)
        plt.show()
