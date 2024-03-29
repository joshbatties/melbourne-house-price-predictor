"""
This module provides functionality to train and evaluate multiple regression models on housing data. 
It is designed to work seamlessly with preprocessed housing data, allowing for a straightforward comparison of 
different models' performance based on the Root Mean Square Error (RMSE) metric.

Example Usage:
    # Assuming `housing_prepared` and `housing_labels` are available and properly prepared
    trainer = ModelTrainer(housing_prepared, housing_labels)
    trainer.train_and_evaluate()
    trainer.display_scores('Random Forest')

This module simplifies the model training and evaluation process, 
making it easy to compare the effectiveness of different regression models on predicting housing prices.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class ModelTrainer:
    """
    A class to train and evaluate different regression models on housing data.

    Attributes:
        housing_prepared (np.array): The preprocessed housing features ready for model training.
        housing_labels (np.array): The actual prices of the housing data.
        models (dict): A dictionary mapping model names to their respective initialized model objects.
        model_scores (dict): A dictionary to store the RMSE scores of the trained models.
    """

    def __init__(self, housing_prepared, housing_labels):
        """
        Initializes the ModelTrainer with the dataset and a set of models.

        Parameters:
            housing_prepared (np.array): The preprocessed features of the housing dataset.
            housing_labels (np.array): The target values (prices) of the housing dataset.
        """

        self.housing_prepared = housing_prepared
        self.housing_labels = housing_labels
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }
        self.model_scores = {}

    def train_and_evaluate(self):
        """
        Trains each model on the prepared housing data and evaluates it using the RMSE metric.
        The results are printed and stored in the `model_scores` attribute.
        """

        for name, model in self.models.items():
            model.fit(self.housing_prepared, self.housing_labels)
            housing_predictions = model.predict(self.housing_prepared)
            mse = mean_squared_error(self.housing_labels, housing_predictions)
            rmse = np.sqrt(mse)
            self.model_scores[name] = rmse
            print(f"{name} trained. RMSE: {rmse}")

    def display_scores(self, model_name):
        """
        Displays the cross-validation scores for a specified model.

        Parameters:
            model_name (str): The name of the model to display scores for.
        """

        scores = cross_val_score(self.models[model_name], self.housing_prepared, self.housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        print(f"Scores for {model_name}: {rmse_scores}")
        print(f"Mean: {rmse_scores.mean()}, Standard deviation: {rmse_scores.std()}")