"""
This module contains the HousingExploration class designed for exploring and visualizing housing data. It provides methods to display basic information about a housing dataset, descriptive statistics, and visualizations including histograms and scatterplots to understand the distribution of data and the relationship between different variables, especially focusing on geographical distribution of prices.

The visualizations are intended to assist in identifying trends, outliers, and patterns in the housing market, using both raw and transformed data for more insightful analyses. Logarithmic transformations are applied where necessary to improve the interpretability of skewed data.

Example usage:
    # Display basic DataFrame information
    HousingExploration.display_basic_dataframe_info(housing_df)

    # Display descriptive statistics of the housing data
    HousingExploration.display_descriptive_statistics(housing_df)

    # Plot histograms for various housing features
    HousingExploration.plot_histograms(housing_df)

    # Plot a scatterplot of housing locations colored by price
    HousingExploration.plot_scatterplot(housing_df
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.ticker import FuncFormatter

class HousingExploration:
    """
    A class for exploring housing data through various visualization and data summary methods.
    
    Methods:
    - display_basic_dataframe_info: Prints the first few rows and info about the DataFrame.
    - display_descriptive_statistics: Prints descriptive statistics of the DataFrame.
    - plot_histograms: Plots histograms for various housing-related features.
    - plot_scatterplot: Plots a scatterplot of housing locations colored by price.
    """

    @staticmethod
    def display_basic_dataframe_info(housing_df):
        """
        Prints the first five rows of the DataFrame and general information about it.
        
        Parameters:
        - housing_df (DataFrame): The housing data to be displayed.
        """

        print(housing_df.head())
        print("\nInfo:")
        housing_df.info()
        

    @staticmethod
    def display_descriptive_statistics(housing_df):
        """
        Prints descriptive statistics for numerical columns in the DataFrame.
        
        Parameters:
        - housing_df (DataFrame): The housing data for which statistics are calculated.
        """

        print("\nDescriptive Statistics:")
        print(housing_df.describe())

    @staticmethod
    def plot_histograms(housing_df):
        """
        Plots histograms for selected numerical columns in the DataFrame, applying log transformation to 'landsize' and 'price' to reduce skewness.
        
        Parameters:
        - housing_df (DataFrame): The housing data for which histograms are plotted.
        """

        # Apply log transformation to 'landsize' and 'price' to reduce skewness, adding 1 to avoid log(0)
        housing_df['log_landsize'] = np.log(housing_df['landsize'] + 1)
        housing_df['log_price'] = np.log(housing_df['price'] + 1)
        
        # List of columns to plot, including the newly transformed ones
        columns_to_plot = ['rooms', 'distance', 'bedrooms', 'bathrooms', 'cars', 'log_landsize', 'lattitude', 'longtitude', 'log_price']
        
        # Number of bins for each histogram: Customized based on the data's nature
        bins_dict = {
            'rooms': 10, 'distance': 20, 'bedrooms': 10, 'bathrooms': 8, 'cars': 10, 
            'log_landsize': 20, 'lattitude': 20, 'longtitude': 20, 'log_price': 20
        }
        
        # Plotting
        plt.figure(figsize=(20, 15))
        for index, column in enumerate(columns_to_plot, 1):
            plt.subplot(3, 3, index)
            housing_df[column].hist(bins=bins_dict[column])
            plt.title(column.replace('_', ' ').capitalize())
        
        plt.suptitle("Improved Histograms of Housing Data Points")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the suptitle
        plt.show()

    @staticmethod
    def plot_scatterplot(housing_df):
        """
        Plots a scatterplot of housing locations, with points colored by the logarithm of the price to highlight differences.
        
        Parameters:
        - housing_df (DataFrame): The housing data for which the scatterplot is plotted.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        # Applying a logarithmic transformation to the price for color mapping
        prices_log = np.log(housing_df["price"])
        scatter = ax.scatter(housing_df["longtitude"], housing_df["lattitude"], alpha=0.4,
                             s=20,  # Using a fixed size for simplicity
                             c=prices_log, cmap=plt.get_cmap("jet"))
        plt.title("Scatterplot of Housing Location and Log-Scaled Price")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Create colorbar with log-scaled labels converted back to price
        colorbar = fig.colorbar(scatter, ax=ax, format=FuncFormatter(lambda x, _: f'${np.exp(x):,.0f}'))
        colorbar.set_label('Price ($)')

        plt.show()

