import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model.model_training import ModelTrainer
from model.model_tuning import ModelTuner
from model.data_preprocessing import HousingPreprocessor
from model.data_load import HousingData


def main():
    # Get user input
    distance = input("Enter the distance from CBD: ")
    bedrooms = input("Enter the number of bedrooms: ")
    bathrooms = input("Enter the number of bathrooms: ")
    cars = input("Enter the number of car spaces: ")
    landsize = input("Enter the land size: ")
    latitude = input("Enter the latitude: ")
    longitude = input("Enter the longitude: ")

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'distance': [distance],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'cars': [cars],
        'landsize': [landsize],
        'latitude': [latitude],
        'longitude': [longitude]
    })

    # Load the housing dataset
    print("Loading data...")
    housing = HousingData.load_housing_data()
    print("Data loaded successfully.")
    # Process the data  
    print("Processing data...")
    preprocessor = HousingPreprocessor(housing)
    preprocessor.bin_suburb_median()
    preprocessor.stratified_split(column='suburb_median', test_size=0.2)
    housing_prepared, housing_labels = preprocessor.prepare_data_for_ml()
    print("Data processed successfully.")

    # Load the final model
    print( "Training model...")
    trainer = ModelTrainer(housing_prepared, housing_labels)
    trainer.train_and_evaluate()
    print("Model trained successfully.")
    print("Tuning model...")
    tuner = ModelTuner(housing_prepared, housing_labels)
    tuner.tune_random_forest()
    print("Random Forest model tuned successfully.")
    final_model = tuner.best_model
    # Preprocess the user data
    user_data_prepared = preprocessor.prepare_data_for_ml(user_data, transform_only=True)

    # Make a prediction
    prediction = final_model.predict(user_data_prepared)
    print(f"The predicted price is: ${round(prediction[0], 2)}")

if __name__ == "__main__":
    main()