"""
This module contains the HousingPreprocessor class, designed to facilitate the preprocessing of housing data. 
It includes functionality for splitting the dataset into stratified training and testing sets, 
and for applying a series of preprocessing steps to the data, including handling missing values and feature scaling.

The preprocessing pipeline is built using scikit-learn's Pipeline and ColumnTransformer classes, 
ensuring a robust and scalable approach to data preparation. 
This module is particularly useful for preparing housing data where numerical features require normalization 
and missing values need to be imputed.

Key Features:
- Stratified splitting of the dataset based on a specified column to ensure representative training and testing sets.
- A preprocessing pipeline that includes median imputation for missing values and standard scaling for numerical features.
- The ability to transform new data using the fitted pipeline, facilitating easy preparation of data for model predictions.


Example Usage:
    # Initialize the preprocessor
    preprocessor = HousingPreprocessor(housing_df)

    # Perform a stratified split based on a specific column
    preprocessor.stratified_split(column='suburb_median', test_size=0.2)

    # Prepare the training data for machine learning
    housing_prepared, housing_labels = preprocessor.prepare_data_for_ml()
"""

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

class HousingPreprocessor:
    def __init__(self, housing_df):
        self.housing_df = housing_df
        self.strat_train_set = None
        self.strat_test_set = None
        self.full_pipeline = None  # Keep the fitted pipeline for transforming test data

    def bin_suburb_median(self):
        """
        Bins the suburb median prices into predefined categories.
        """
        bins = [0, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, np.inf]
        labels = ['0-500k', '500k-600k', '600k-700k', '700k-800k', '800k-900k', '900k-1M', '1M-1.1M', '1.1M-1.2M', '1.2M-1.3M', '1.3M-1.4M', '1.4M-1.5M', '1.5M+']
        self.housing_df['suburb_median'] = pd.cut(self.housing_df['suburb_median'], bins=bins, labels=labels)

    def stratified_split(self, column, test_size=0.2):
        """
        Splits the dataset into a stratified training set and test set based on the specified column.
        
        Parameters:
        - column (str): The column name to use for stratified splitting.
        - test_size (float): The proportion of the dataset to include in the test split.
        """
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_index, test_index in split.split(self.housing_df, self.housing_df[column]):
            self.strat_train_set = self.housing_df.loc[train_index]
            self.strat_test_set = self.housing_df.loc[test_index]

    def prepare_data_for_ml(self, data=None, transform_only=False):
        """
        Prepares housing data for machine learning by applying preprocessing steps.
        
        Parameters:
        - data (DataFrame, optional): The dataset to be prepared. If None, uses the stratified training set.
        - transform_only (bool): If True, transforms the data using the existing pipeline without fitting.
        
        Returns:
        - A tuple of the prepared features and the labels if transforming the training set; otherwise, just the prepared features.
        """
        if data is None:
            data = self.strat_train_set
        
        if not transform_only:
            housing = data.drop("price", axis=1)
            housing_labels = data["price"].copy()

            num_attribs = list(housing.select_dtypes(include=[np.number]))

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('std_scaler', StandardScaler()),
            ])

            self.full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
            ])

            housing_prepared = self.full_pipeline.fit_transform(housing)
            return housing_prepared, housing_labels
        else:
            # Transforming data without fitting; used for the test set
            return self.full_pipeline.transform(data)
