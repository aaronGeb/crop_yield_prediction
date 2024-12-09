#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional


class DataProcessing:
    def __init__(self, data: DataFrame = None):
        self.data = data

    def read_data(self, file_path: str) -> DataFrame:
        """load data from csv file
        Args:
            file_path: path to the csv file
        Returns:
            data: DataFrame
        """
        self.data = pd.read_csv(file_path)
        return self.data

    def remove_duplicates(self) -> DataFrame:
        """remove duplicates from the data
        Returns:
            data: DataFrame
        """
        self.data = self.data.drop_duplicates()
        return self.data

    def standardize_columns_names(self) -> DataFrame:
        """standardize the data
        Args:
            columns: list of columns to standardize
        Returns:
            data: DataFrame
        """
        self.data = self.data.rename(
            columns=lambda col: col.strip()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        return self.data

    def head(self, n: Optional[int] = 5) -> DataFrame:
        """return the first n rows of the data
        Args:
            n: number of rows to return
        Returns:
            data: DataFrame
        """
        return self.data.head(n).T
    def 
