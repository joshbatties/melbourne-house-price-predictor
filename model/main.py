import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_load import HousingData

def main():

    # Load the housing dataset
    housing = HousingData.load_housing_data()
    print(housing.head()) 

if __name__ == "__main__":
    main()
