# Sales Forecasting Using Predictive Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("C:/Users/SABIC-Recreation/Downloads/superstore_final_dataset (1).csv", encoding='latin1')

df.head()
df.shape
df.info()
df.isnull().sum()

df.drop_duplicates(inplace=True)
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True)

monthly_sales = df.groupby(df['Order_Date'].dt.to_period('M'))['Sales'].sum()
monthly_sales = monthly_sales.reset_index()
monthly_sales['Order_Date'] = monthly_sales['Order_Date'].astype(str)
monthly_sales['Order_Date'] = pd.to_datetime(monthly_sales['Order_Date'])

plt.figure(figsize=(12,6))
plt.plot(monthly_sales['Order_Date'], monthly_sales['Sales'])
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

monthly_sales['Month_Number'] = np.arange(len(monthly_sales))
X = monthly_sales[['Month_Number']]
y = monthly_sales['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(10,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

future_months = pd.DataFrame({'Month_Number': np.arange(len(monthly_sales), len(monthly_sales)+12)})
future_predictions = model.predict(future_months)

plt.figure(figsize=(12,6))
plt.plot(monthly_sales['Month_Number'],monthly_sales['Sales'],label='Historical Sales')
plt.plot(future_months['Month_Number'],future_predictions,label='Forecasted Sales')
plt.legend()
plt.title("Sales Forecast")
plt.show()
