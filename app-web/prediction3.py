import pandas as pd
from joblib import dump,load
import numpy as np

######### Prediction 3 ###################
def prediction3LinearRegression():
    df = pd.read_csv('test.csv')
    X = df['G1']
    Y=  df['G3']
    lin_reg3 = load('./Model3/linreg_model3.pkl')
    s = lin_reg3.predict(X)
    print(s)
    return s;
