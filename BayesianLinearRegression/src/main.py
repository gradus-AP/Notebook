import numpy as np
from LinearRegression import LinearRegression
import pandas as pd
"""
Test data (3 attributions and 20 samples) 
Z | x_0, x_1 ~ Normal(1.0 + 3.0 * x_0 + 0.70 * x_1, 1.0)
"""
df = pd.read_csv('.\\.\\test_data.csv')

model = LinearRegression(2, 0.0, 0.1, np.array([0.0, 0.0]), np.identity(2), 0.1, 0.1)
model._fit(df.col1,df.loc[:,'col2':'col3'], 0.0, 0.1, 0.1, 0.1, iteration = 10)

print("bias : \n",model.post.mu,"coefficients : \n",model.post.beta)
model.plotHyperParams()