#######################################################################
#                                                                     #
#            Pima Indians Diabetes Dataset Analysis                   #
#                                                                     #
# Authors: Augustin Chavanes & Corentin Salvi                         #
#######################################################################
import pandas as pd 
import matplotlib.pyplot as plt
from pandas import read_csv
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

diabetes=pd.read_csv('./train.csv',na_values=['?'])
print(diabetes.describe())
print(diabetes.info())
print(diabetes.isnull().sum())
print(diabetes.tail())

for column in diabetes.columns:
    plt.figure(figsize=(8, 4))
    diabetes[column].hist(bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()