# Author: Daniel Kennett
import pandas as pd
import numpy as np
# import torch

fama_french = pd.read_csv('data/F-F_Research_Data_5_Factors_2x3.csv')
fama_french.rename({'Unnamed: 0':'date'}, axis=1, inplace=True)
print(fama_french)

predictor_factors = pd.read_excel('data/PredictorData2023.xlsx')
predictor_factors.rename({'yyyymm':'date'}, axis=1, inplace=True)

print(predictor_factors)