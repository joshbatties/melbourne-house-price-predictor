

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_fetch_load import HousingData

def main():

    # Fetch and load the data
    HousingData.fetch_housing_data()
    housing = HousingData.load_housing_data()

if __name__ == "__main__":
    main()
