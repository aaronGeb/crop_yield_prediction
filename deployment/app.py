import os
import sys
import pickle
import streamlit as st
import pandas as pd


# Set up paths
sys.path.append(os.path.abspath(os.path.join("../scripts")))
root_path = os.path.abspath("..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from predict import CropPredictor


soil_type_mapping = {
    "Loamy": 0,
    "Peaty": 1,
    "Rocky": 2,
    "Sandy": 3,
    "Silt": 4,
    "Volcanic": 5,
}
crop_type_mapping = {
    "Banana": 0,
    "Cassava": 1,
    "Coffee": 2,
    "Maize": 3,
    "Potato": 4,
    "Rice": 5,
    "Tea": 6,
    "Wheat": 7,
}


class CropYieldApp:
    def __init__(self):
        # Initialize the CropPredictor and load the model
        self.model_path = os.path.join(root_path, "models/random_forest_model.pkl")

        # Ensure the predictor is initialized in session_state
        if "predictor" not in st.session_state:
            st.session_state.predictor = CropPredictor(model_path=self.model_path)
            st.session_state.predictor.load_model()  # Only load once
            st.session_state.model_loaded = (
                True  # Flag indicating the model has been loaded
            )
        else:
            # Ensure that the model is only loaded if not already in session_state
            if (
                not hasattr(st.session_state.predictor, "model")
                or st.session_state.predictor.model is None
            ):
                st.session_state.predictor.load_model()
                st.session_state.model_loaded = True

    def get_user_input(self):
        """Get user input for the crop prediction features."""
        st.title("Crop Yield Prediction")
        st.write("Enter the features to predict the crop yield.")

        # Input fields for user to provide feature data
        elevation = st.number_input(
            "Elevation", min_value=0, max_value=10000, value=500
        )
        latitude = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0, value=9.678
        )
        longitude = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0, value=45.123
        )
        slope = st.number_input("Slope", min_value=0, max_value=90, value=15)
        rainfall = st.number_input("Rainfall", min_value=0, max_value=5000, value=300)
        min_temperature_c = st.number_input(
            "Min Temperature (°C)", min_value=-50, max_value=50, value=10
        )
        max_temperature_c = st.number_input(
            "Max Temperature (°C)", min_value=-50, max_value=50, value=35
        )
        ave_temps = st.number_input(
            "Average Temperature (°C)", min_value=-50, max_value=50, value=22
        )
        soil_type = st.selectbox(
            "Soil Type",
            options=["Loamy", "Peaty", "Sandy", "Silt", "Volcanic"],
            index=0,
        )
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
        # Display categorical variables

        crop_type = st.selectbox(
            "Crop Type",
            options=[
                "Banana",
                "Cassava",
                "Coffee",
                "Maize",
                "Potato",
                "Rice",
                "Tea",
                "Wheat",
            ],
            index=0,
        )
        # Map the user input to numeric values for model prediction
        soil_type_value = soil_type_mapping[soil_type]
        crop_type_value = crop_type_mapping[crop_type]
        # Create a DataFrame with input features
        input_data = pd.DataFrame(
            {
                "elevation": [elevation],
                "latitude": [latitude],
                "longitude": [longitude],
                "slope": [slope],
                "rainfall": [rainfall],
                "min_temperature_c": [min_temperature_c],
                "max_temperature_c": [max_temperature_c],
                "ave_temps": [ave_temps],
                "soil_type": [soil_type_value],
                "ph": [ph],
                "crop_type": [crop_type_value],
            }
        )

        return input_data

    def predict_yield(self, input_data):
        """Make a crop yield prediction using the provided input data."""
        if st.button("Predict"):
            try:
                if st.session_state.model_loaded:
                    predictions = st.session_state.predictor.predict_yield(input_data)
                    if predictions is None or len(predictions) == 0:
                        st.error(
                            "Prediction could not be made. Please check the input data."
                        )
                    else:
                        st.write(f"Predicted Crop Yield: {predictions[0]:.4f}")
                else:
                    st.error("Model not loaded. Please try again.")
            except (ValueError, AttributeError) as e:
                st.error(f"Error during prediction: {e}")

    def run(self):
        """Run the app: collect input data and make predictions."""
        input_data = self.get_user_input()
        self.predict_yield(input_data)


# Main function to run the app
if __name__ == "__main__":
    app = CropYieldApp()
    app.run() 
