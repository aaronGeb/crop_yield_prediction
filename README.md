# Crop Yield Prediction

## Overview

The **Crop Yield Prediction** application leverages machine learning to predict crop yields based on various environmental factors. Using a Random Forest Regressor model, this application can predict the yield of crops such as Banana, Maize, Rice, and others, given the input features like elevation, latitude, rainfall, and temperature.
### Features 
- Input Features
    - Elevation
    - Latitude
    - Longitude
    - Slope
    - Rainfall
    - Minimum Temperature
    - Maximum Temperature
    - Average Temperature
    - Soil Type (Categorical: Loamy, Peaty, Sandy, Silt, Volcanic)
    - Ph (Acidity level)
    - Crop Type (Categorical: Banana, Cassava, Coffee, Maize, Potato, Rice, Tea, Wheat)
- Model: The application uses a RandomForestRegressor model trained on historical data to predict crop yield based on the provided input features

## Installation
**Prerequisites** 

Ensure you have the following installed:
- Python 3.9+
- Streamlit for the web UI
- pandas for data manipulation
- scikit-learn for machine learning

### Setup
1. Clone the repository
```
git clone https://github.com/aarongeb/crop_yield_prediction.git
```
2. 	Navigate to the project directory:
```
cd crop_yield_prediction
```
3.	Install the required dependencies:
```
conda install --file environmental.yaml
```
4.	Run the application:
```
streamlit run app.py
```
This will launch the app on http://localhost:8501

### Usage

### Web Interface

1.Upon launching the app, you will see a form where you can enter the following:
- Geographical Information: Elevation, latitude, longitude, and slope.
- Climate Data: Rainfall, minimum/maximum/average temperature, and pH.
- Soil Information: Type of soil and crop type.
  
2.Once the data is entered, click Predict to see the predicted crop yield for the provided inputs.

### Example Input:

- Elevation: 670.77900
- Latitude: -14.742861
- Longitude: -6.1102221
- Slope: 7.636470
- Rainfall: 1568.60
- Min Temperature: -3.8°C
- Max Temperature: 33.4°C
- Average Temperature: 14.80°C
- Soil Type: volcanic
- pH: 5.385873
- Crop Type: tea
  
![Crop Yield Prediction ](./image/Screenshot%202024-12-12%20at%2013.48.47.png)

### Model Explanation

The underlying model used for prediction is a Random Forest Regressor from the scikit-learn library. The model was trained on a dataset containing environmental factors and corresponding crop yields. Random Forest is an ensemble learning method that builds multiple decision trees and combines their results for more accurate predictions.

### Files

- deployment/app.py: The Streamlit app file that contains the web interface and interacts with the model.
- models/random_forest_model.pkl: The trained Random Forest model.
- scripts/predict.py: Contains the CropPredictor class used to load the model and make predictions.
- requirements.txt: Lists the required dependencies for the project.
  
### Contributing

If you’d like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request. Please ensure your contributions are well-documented and tested.

### License

This project is licensed under the MIT License - see the [LICENSE]() file for details.
