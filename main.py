#Importing Data and Content of it.
import pandas as pd
real_estate_data = pd.read_csv("Real_Estate.csv")
real_estate_data_head = real_estate_data.head()
data_info = real_estate_data.info()
print(real_estate_data_head)
print(data_info)

#Looking in data if it is containing any NULL Values
print(real_estate_data.isnull().sum())

#Looking for Descriptive Statistics of the Dataset
descriptive_stats = real_estate_data.describe()
print(descriptive_stats)

#Looking for the Histogram of all the Numerical Features
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
fig, axis = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
fig.suptitle('Histograms of Real Estate Data', fontsize=16)

cols = ['House age','Distance to the nearest MRT station','Number of convenience stores','Latitude','Longitude','House price of unit area']

for i, col in enumerate(cols):
    sns.histplot(real_estate_data[col], kde=True, ax=axis[i//2, i%2])
    axis[i//2, i%2].set_title(col)
    axis[i//2, i%2].set_xlabel('')
    axis[i//2, i%2].set_ylabel('')

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

#Creating Scatter Plots to find Relationship between these Variables and the House Price
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)
sns.scatterplot(data=real_estate_data, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(data=real_estate_data, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(data=real_estate_data, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(data=real_estate_data, x='Latitude', y='House price of unit area', ax=axes[1, 1])

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

#Correlation Matrix
correlation_matrix = real_estate_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix)

#Building Regression Model to predict the Real Estate PRices by using Linear Regression Algorithm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

features = ['Distance to the nearest MRT station', 'Number of convenience stores','Latitude','Longitude']
target ='House price of unit area'

x = real_estate_data[features]
y = real_estate_data[target]

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

#Visualize the Actual Vs Predicted Values to Assess how well model is performing
y_pred_lr = model.predict(x_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual Vs Predicted House Prices')
plt.show()

