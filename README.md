# Paddy Crop Analysis

A data analysis and machine learning project for paddy crop datasets, developed in Python.  
This project explores crop-related variables through preprocessing, statistical analysis, visualization, dimensionality reduction, and predictive modelling.

## Project Overview

The purpose of this project is to analyse paddy crop data and extract useful insights for understanding patterns, relationships, and predictive factors within the dataset.  
The workflow combines data cleaning, exploratory data analysis, feature transformation, and machine learning models in order to provide both descriptive and predictive perspectives.

This project includes:

- data preprocessing
- univariate analysis
- bivariate analysis
- PCA-based dimensionality reduction
- growth stage analysis
- linear regression prediction
- random forest prediction
- an integrated main application

## Project Structure

```text
Paddy-crop-analysis/
│
├── assets/
│   ├── Efrei-Logo.png
│   └── WUT-Logo.png
│
├── data/
│   └── paddydataset.csv
│
├── src/
│   ├── PCA_analysis.py
│   ├── PCA_preprocessing.py
│   ├── acknowledgements.py
│   ├── bivariate_analysis.py
│   ├── data_preprocessing.py
│   ├── growth_stage_analysis.py
│   ├── linear_regression_prediction.py
│   ├── main_app.py
│   ├── random_forest_prediction.py
│   └── univariate_analysis.py
│
├── requirements.txt
├── .gitignore
└── README.md