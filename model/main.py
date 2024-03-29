import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_load import HousingData
from data_explore import HousingExploration
from data_preprocessing import HousingPreprocessor
from model_training import ModelTrainer
from model_tuning import ModelTuner

def main():

    # Load the housing dataset
    housing = HousingData.load_housing_data()

    # Data exploration (optional, for insight)
    HousingExploration.display_basic_dataframe_info(housing) 
    HousingExploration.display_descriptive_statistics(housing)
    HousingExploration.plot_histograms(housing)
    HousingExploration.plot_scatterplot(housing)

    # Initialize the preprocessor
    preprocessor = HousingPreprocessor(housing)

    # Bin suburb median prices into categories for stratified sampling
    bins = [0, 500000, 1000000, 1500000, 2000000, np.inf]
    labels = ['0-500k', '500k-1M', '1M-1.5M', '1.5M-2M', '2M+']
    housing['suburb_median'] = pd.cut(housing['suburb_median'], bins=bins, labels=labels)

    # Perform a stratified split based on the 'suburb_median' column
    preprocessor.stratified_split(column="suburb_median", test_size=0.2)

    # Prepare the training data
    housing_prepared, housing_labels = preprocessor.prepare_data_for_ml()

    # Model training and evaluation
    trainer = ModelTrainer(housing_prepared, housing_labels)
    trainer.train_and_evaluate()
    
    # Display the Cross Validation scores for each model
    trainer.display_scores('Linear Regression')
    trainer.display_scores('Decision Tree')
    trainer.display_scores('Random Forest')

    # Hyperparameter tuning for the best model (Random Forest here as an example)
    tuner = ModelTuner(housing_prepared, housing_labels)
    tuner.tune_random_forest()

    # Prepare the test set
    X_test = preprocessor.strat_test_set.drop("price", axis=1)
    y_test = preprocessor.strat_test_set["price"].copy()
    X_test_prepared = preprocessor.prepare_data_for_ml(X_test, transform_only=True)

    # Final model evaluation on the test set
    tuner.evaluate_on_test_set(X_test_prepared, y_test)


if __name__ == "__main__":
    main()
