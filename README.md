# melbourne-house-price-predictor
This project is supervised machine learning model that predicts the sale price of houses in Melbourne, Australia. 

The main steps of the project are:

    1. Load the housing dataset.

    2. Explore the data to gain insights. This includes displaying basic information about the dataset, descriptive statistics, and visualizations such as histograms and scatter plots.

    3. Preprocess the data for machine learning. This includes binning the 'suburb_median' prices into categories for stratified sampling, performing a stratified split based on the 'suburb_median' column to create a training set and a test set, and preparing the training data.

    4. Train and evaluate machine learning models. This includes training a Linear Regression model, a Decision Tree model, and a Random Forest model, and evaluating their performance using cross-validation.

    5. Tune the hyperparameters of the best model (in this case, the Random Forest model) to improve its performance.

    6. Evaluate the final model on the test set.

You can use the model by running the script and following the prompts: 
>>> python predict.py

# Data
The input dataset is melb_housing.csv created by Joshua Batties

It contains 9399 instances of a sale of a house in Melbourne between 2016-2018.
Each instance has a value associated with the:
    - suburb median price 
    - distance from CBD	
    - number of bedrooms	
    - number of bathrooms	
    - number of car spaces
    - land size	
    - lattitude	
    - longtitude
    - price

This dataset is a snapshot of a dataset from  https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
Which is based on scraped data from Domain.com.au between 2016-2018
I have removed the suburb, address, method, seller, date, postcode, building area, year built, council, region and property count columns.

I have removed all rows all instances of "unit" or "townhouse" so my model only outputs predictions for houses.

I also created a column for the median house price in the suburb. 
Median House Prices were taken for each suburb in 2017 from https://discover.data.vic.gov.au/dataset/victorian-property-sales-report-median-house-by-suburb-time-series

All the suburb names in the original dataset are replaced with the median house price for that suburb in 2017.
Then this is converted to a categorical attribute which has ranges for the median price in $100,000 intervals.

TODO:

- Input validation for the predict.py
- Research potenital solutions for allowing user to input a address and that will then return all the inputs for the predict.py




