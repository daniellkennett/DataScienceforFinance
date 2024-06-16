"""
Author: Daniel Kennett
The point of this script is for data cleanup.
Output should be a csv or xslx for ML processing
"""
import pandas as pd
import numpy as np


def clean_fama_french():
    fama_french = pd.read_csv('data/F-F_Research_Data_5_Factors_2x3.csv')
    fama_french.rename({'Unnamed: 0':'date'}, axis=1, inplace=True)
    fama_french.set_index('date', inplace=True)

    # Divide all columns by 100 to normalize as percents
    for col in fama_french.columns: 
        fama_french[col] = fama_french[col]/100
    return fama_french


def clean_predict_factors():
    predictor_factors = pd.read_excel('data/PredictorData2023.xlsx')
    predictor_factors.rename({'yyyymm':'date'}, axis=1, inplace=True)
    predictor_factors.set_index('date', inplace=True)

    # csp isn't tracked anymore, so we drop
    predictor_factors.drop('csp', inplace=True, axis=1)
    # Some other factors weren't tracked until ~1920s. Drop everything before that.
    predictor_factors.dropna(inplace=True)

    # Find change in index
    predictor_factors['delta_Index'] = predictor_factors['Index'].pct_change()
    return predictor_factors



if __name__ == "__main__":
    fama_french = clean_fama_french()
    predictor_factors = clean_predict_factors()
    
    output_data = fama_french.join(predictor_factors, how='left')
    # print(output_data)
    
    output_data.to_csv('data/CleanData.csv')
