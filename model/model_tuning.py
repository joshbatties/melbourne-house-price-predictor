"""
This module provides the ModelTuner class, which facilitates hyperparameter tuning for regression models using grid search. 
It is specifically designed to fine-tune a RandomForestRegressor model 
by exploring a predefined grid of parameters and evaluating model performance through cross-validation.

Example Usage:
    # Assuming `housing_prepared` and `housing_labels` are ready
    tuner = ModelTuner(housing_prepared, housing_labels)
    tuner.tune_random_forest()
    # Assuming `X_test_prepared` and `y_test` are the prepared test set and labels
    tuner.evaluate_on_test_set(X_test_prepared, y_test)
"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy import stats

class ModelTuner:
    """
    A class to tune RandomForestRegressor models and evaluate them on a test set.
    
    Attributes:
        housing_prepared (np.array): The preprocessed feature set for training.
        housing_labels (np.array): The target values for training.
        best_model (estimator): The best model found by GridSearchCV.
    """

    def __init__(self, housing_prepared, housing_labels):
        """
        Initializes the ModelTuner with training data.
        
        Parameters:
            housing_prepared (np.array): The preprocessed features of the housing dataset.
            housing_labels (np.array): The target values (prices) of the housing dataset.
        """

        self.housing_prepared = housing_prepared
        self.housing_labels = housing_labels
        self.best_model = None

    def tune_random_forest(self):
        """
        Performs grid search to find the best hyperparameters for the RandomForestRegressor.
        The results are stored in `best_model`.
        """

        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(self.housing_prepared, self.housing_labels)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

    def evaluate_on_test_set(self, X_test_prepared, y_test):
        """
        Evaluates the performance of the best model (found through grid search) on the test set.

        This method predicts housing prices using the best model on the provided test set features,
        computes the RMSE to quantify prediction errors, and calculates a 95% confidence interval for
        the RMSE, offering insights into the precision of the model's performance estimate.

        Parameters:
          X_test_prepared (np.array): The preprocessed feature set for testing. This should be 
                                    prepared in the same way as the training set to ensure 
                                    consistency in input data format.
          y_test (np.array): The actual target values for testing. These are the true housing prices 
                           for the test set, used to evaluate the accuracy of the model's predictions.

        Prints:
          - The RMSE of the model's predictions on the test set, providing a single measure of 
            prediction error across all test instances.
          - A 95% confidence interval for the RMSE, indicating the range within which the true RMSE 
            is likely to lie with 95% certainty, assuming the test set is a typical sample from the 
            population of interest.
        """
        
        final_predictions = self.best_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        print(f"Final RMSE on test set: {final_rmse}")
        
        # Confidence interval calculation
        confidence = 0.95
        squared_errors = (final_predictions - y_test) ** 2
        confidence_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                                       loc=squared_errors.mean(),
                                                       scale=stats.sem(squared_errors)))
        print(f"95% confidence interval for the RMSE: {confidence_interval}")