# melbourne-house-price-predictor
This supervised machine learning model takes in features about a house in Melbourne and predicts the sale price.

# Data
The input data  is a csv file created by Joshua Batties

It contains 9399 instances of a sale of a house in Melbourne between 2016-2018.
Each instance has a value associated with the:
    - suburb median price 
    - number of rooms	
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
Median House Prices were taken for each suburb in 2017 from
https://discover.data.vic.gov.au/dataset/victorian-property-sales-report-median-house-by-suburb-time-series

I then used excel to replace all the suburb names in the original dataset with the median house price for that suburb in 2017.





