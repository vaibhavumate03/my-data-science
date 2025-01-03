# Analysis using regression

import sklearn 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame ({
    'petrol':[100,90,80,75,70,67,62,57,50,45],
     'milk':[40,36,33,30,27,25,24,22,20,18]
})

x=data[['petrol']]
y=data['milk']

x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print ("MSE",mean_squared_error(y_test,y_pred))
print ("R2-error",r2_score(y_test,y_pred))

newmilk = model.predict([[105],[40]])
newmilk
