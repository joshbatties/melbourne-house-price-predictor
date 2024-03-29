import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_load import HousingData
from data_explore import HousingExploration

def main():

    # Load the housing dataset
    housing = HousingData.load_housing_data()

    # Data exploration (optional, for insight)
    HousingExploration.display_basic_dataframe_info(housing) 
    HousingExploration.display_descriptive_statistics(housing)
    HousingExploration.plot_histograms(housing)
    HousingExploration.plot_scatterplot(housing)

if __name__ == "__main__":
    main()
