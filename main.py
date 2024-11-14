import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("satislar.csv")

data_x_axis = data.iloc[:,0:1]
data_y_axis = data.iloc[:,1:]

x_train, x_test, y_train, y_test = train_test_split(data_x_axis, data_y_axis, test_size=0.33, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, model.predict(x_test))
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()


print("-"*50)
print(r2_score(y_test, y_pred)*100)