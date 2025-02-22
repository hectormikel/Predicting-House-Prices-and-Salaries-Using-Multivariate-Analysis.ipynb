House Price and Salary Prediction with Multiple Variables
This repository contains Python code for predicting house prices and salaries using multiple variables. The code leverages the polars library for data manipulation and scikit-learn for linear regression modeling.

Table of Contents

Introduction
Installation
Usage
House Price Prediction
Salary Prediction
Dependencies
License
Introduction

The project demonstrates how to use linear regression to predict:

House Prices: Based on features like area, number of bedrooms, and age of the house.
Salaries: Based on experience, test scores, and interview scores.
The code handles missing data by filling null values with appropriate defaults (e.g., median for numerical columns and 'zero' for experience).

Installation

To run this code, you need to install the required dependencies. You can do this using pip:

bash
Copy
pip install polars pandas scikit-learn matplotlib
Usage

House Price Prediction

The script reads a CSV file (homeprices_mult.csv) containing house price data, preprocesses it, and trains a linear regression model to predict house prices.

python
Copy
import polars as pl
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import math

# Load data
df = pl.read_csv('homeprices_mult.csv')

# Fill null values in 'bedrooms' with the median
median_bedrooms = math.floor(df['bedrooms'].median())
df = df.with_columns([
    pl.col('bedrooms').fill_null(median_bedrooms)
])

# Prepare features and target
X = df.select([pl.col('area'), pl.col('bedrooms'), pl.col('age')])
y = df.select(pl.col('price'))

# Train the model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Predict house prices
print(reg.predict([[3000, 3, 40]]))  # Predict for 3000 sqft, 3 bedrooms, 40 years old
print(reg.predict([[2500, 4, 5]]))   # Predict for 2500 sqft, 4 bedrooms, 5 years old
Salary Prediction

The script reads a CSV file (hiring.csv) containing hiring data, preprocesses it, and trains a linear regression model to predict salaries.

python
Copy
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

# Load data
df = pl.read_csv('hiring.csv')

# Fill null values in 'experience' with 'zero'
df = df.with_columns(
    pl.col('experience').fill_null('zero')
)

# Fill null values in 'test_score(out of 10)' with the median
df = df.with_columns(
    pl.col('test_score(out of 10)').fill_null(math.floor(df['test_score(out of 10)'].median()))
)

# Map experience values to integers
experience_mapping = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11
}

df = df.with_columns(
    pl.col('experience').replace(experience_mapping, default=None).cast(pl.Int64)
)

# Prepare features and target
X = df.select([pl.col('experience'), pl.col('test_score(out of 10)'), pl.col('interview_score(out of 10)')])
y = df.select(pl.col('salary($)'))

# Train the model
model = linear_model.LinearRegression()
model.fit(X, y)

# Predict salaries
print(model.predict([[2, 9, 6]]))   # Predict for 2 years experience, test score 9, interview score 6
print(model.predict([[12, 10, 10]]))  # Predict for 12 years experience, test score 10, interview score 10
Dependencies

Python 3.x
polars
pandas
scikit-learn
matplotlib
