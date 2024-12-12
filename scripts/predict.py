#!/usr/bin/env python
import pickle
import pandas as pd
from pandas import DataFrame


class CropPredictor:
    """Class for loading the model and predicting the crop yield"""

    def __init__(self, model_path: str):
        """
          Initializes the CropPredictor class.
        Args:
            model_path (str): Path to the trained model file (pickle format)
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Loads the trained model from the specified path.

        Returns:
            The loaded model.
        """
        print(f"Loading model from: {self.model_path}")
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)  # Assign to self.model
            print("Model loaded successfully")
            print(f"Our model is: {self.model}")

        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            raise

    def predict_yield(self, input_data: DataFrame):
        """
        Makes predictions on the input data using the loaded model.

        Args:
            input_data: The input features for making predictions.

        Returns:
            The predicted values.
        """
        if self.model is None:
            print("Model is not loaded. Please load the model first.")
            return None

        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            print(f"An error occurred while making predictions: {e}")
            raise


if __name__ == "__main__":
    predictor = CropPredictor(model_path="../models/random_forest_model.pkl")
    predictor.load_model()

    new_data = pd.DataFrame(
        {
            "elevation": [670.77900],
            "latitude": [-14.742861],
            "longitude": [-6.110221],
            "slope": [7.636470],
            "rainfall": [1586.60],
            "min_temperature_c": [-3.8],
            "max_temperature_c": [33.4],
            "ave_temps": [14.80],
            "soil_type": [5],
            "ph": [5.385873],
            "crop_type": [6],
        }
    )

    predictions = predictor.predict_yield(new_data)
    if predictions is not None:
        print(f"Predictions: {predictions}")
