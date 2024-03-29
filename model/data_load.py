"""
This module defines the HousingData class used for loading housing data from a CSV file into a pandas DataFrame. 

Example:
    housing_data = HousingData.load_housing_data()
    print(housing_data.head())

Note:
    The dataset path and filename are set as class attributes and can be overridden if the dataset is stored in a different location or if a different dataset is to be used.
"""
import os
import pandas as pd

class HousingData:
    """
    A class used to load housing data from a CSV file.
    
    Attributes
    ----------
    HOUSING_PATH : str
        The default path to the directory containing the housing data file.
    HOUSING_FILE : str
        The name of the CSV file containing the housing data.
    """

    HOUSING_PATH = "datasets"
    HOUSING_FILE = "melb_housing.csv"

    @staticmethod
    def load_housing_data(housing_path=HOUSING_PATH, housing_file=HOUSING_FILE):
        """
        Load housing data from a CSV file into a pandas DataFrame.
        
        Parameters
        ----------
        housing_path : str, optional
            The path to the directory containing the housing data file, by default HOUSING_PATH
        housing_file : str, optional
            The name of the CSV file to load, by default HOUSING_FILE

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the loaded housing data.
        """

        csv_path = os.path.join(housing_path, housing_file)
        return pd.read_csv(csv_path)
