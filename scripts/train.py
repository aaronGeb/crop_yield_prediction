#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional
from sklearn.preprocessing import LabelEncoder
import pickle
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RandomForestTrainer:
    """model for crop prediction"""

    def __init__(self, data_path: str, model_path="../models/random_forest_model.pkl"):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> DataFrame:
        """load data from csv file
        Args:
          file_path: path to the csv file
        Returns:
          data: DataFrame
        """
        self.data = pd.read_csv(self.data_path)
        return self.data

    def split_and_prepare_data(self):
        """
        Splits the dataset into training and testing sets,
         separates the target column,and resets the index.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        # Split the dataset into training and testing sets
        self.X_train, self.X_test = train_test_split(
            self.data, test_size=0.3, random_state=42
        )

        # Reset the index for both splits
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)

        # Extract target values
        self.y_train = self.X_train["standard_yield"].values
        self.y_test = self.X_test["standard_yield"].values

        # Remove the target column from features
        self.X_train = self.X_train.drop(columns=["standard_yield"])
        self.X_test = self.X_test.drop(columns=["standard_yield"])

    def train_model(self):
        """train the model"""
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model trained successfully")

    def evaluate_model(self):
        """evaluate the model"""
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        print("Model has a R^2 score of {:.3f}".format(r2))
        print("Model has a MSE score of {:.3f}".format(mse))
        print("Model has a MAE score of {:.3f}".format(mae))

    def save_model(self):
        """Save the model to the specified path."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as file:
            pickle.dump(self.model, file)
        print("Model saved successfully")

    def run(self):
        """Execute the full training pipeline."""
        data = self.load_data()
        self.split_and_prepare_data()
        self.train_model()
        self.evaluate_model()
        self.save_model()


if __name__ == "__main__":
    trainer = RandomForestTrainer(
        data_path="../data/labeled/labeled_agric_survey_data_cleaned.csv"
    )
    trainer.run()
