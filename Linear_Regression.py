import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.float_format',lambda x: '%.2f' %x)

# Simple Linear Regression


df = pd.read_csv("datasets/Advertising.csv")
df.shape
df.head()

X = df[["TV"]]
y = df[["Sales"]]

# Model
simple_reg_model = LinearRegression().fit(X,y)

# sabit (b - bias)
simple_reg_model.intercept_[0]

# tv'nin katsayısı (w1)
simple_reg_model.coef_[0][0]

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(simple_reg_model.intercept_[0], 2)} + TV*{round(simple_reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

#Model_Değerlendirme
y_pred = simple_reg_model.predict(X)
mean_squared_error(y, y_pred)

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-KARE
simple_reg_model.score(X, y)


# Multiple Linear Regression

df = pd.read_csv("datasets/advertising.csv")
X = df.drop('Sales', axis=1)
y = df[["Sales"]]


# Model
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)

np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))


