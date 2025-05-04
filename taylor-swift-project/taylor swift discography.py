'''
Dataset: Taylor Swift Discography
Source: Kaggle
URL: https://www.kaggle.com/datasets/delfinaoliva/taylor-swift-discography
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split

path = r"C:\Users\0753780\Downloads\taylor_swift_discography_updated.csv"
disc_data = pd.read_csv(path, sep=';') #separates the columns by ';'
            
independent = ['tempo']
dependent = 'liveness'
random_state = 42 #original: 42
test_size = 0.5 #original: 0.2

def clean_data():
    global disc_data
    keys = ['spotify_streams', 'videoclip_views','album_physical_sales']
    for key in keys:
        disc_data[key] = disc_data[key].str.replace('.', '')

clean_data()
print("[MODEL]: DATA READY TO BE PROCESSED.")
X = disc_data[independent]
Y = disc_data[dependent]
print("[MODEL]: SPLITTING DATA...")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
model = LinearRegression()
model.fit(X_train, y_train)
print("[MODEL]: PREDICTING...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("[MODEL]: FINISHED CALCULATING ACCURACY.")
factors = independent[0]

title = f"Taylor Swift Discography\nX: {factors} | Predicting: {dependent} | RMSE: {rmse:.2f}"
x_label = f'Actual {dependent}'
y_label = f'Predicted {dependent}'

print("[MODEL]: PREPARING TO SHOW PLOT....")
plt.scatter(X_test, y_test, color='blue', label=x_label)
plt.plot(X_test, y_pred, color='red', label=y_label)
plt.xlabel(factors)
plt.ylabel(dependent)
plt.title(title)
plt.legend()
plt.show()
print("[MODEL]: PLOT SUCCESSFULLY SHOWN.")
print("-------------------------------")
predictions = pd.DataFrame({f'Actual {dependent}': y_test, f'Predicted {dependent}': y_pred})
print(predictions.head())
print(f"ROOT MEAN SQUARE ERROR (RMSE): {rmse}")
print("-------------------------------")
#multiple labels this time
X_multi = disc_data[['tempo', 'loudness', 'energy', 'valence', 'acousticness']]
# Split the data again
X_train, X_test, y_train, y_test = train_test_split(X_multi, Y, test_size=test_size, random_state=random_state)
# Train the model with multiple features
model.fit(X_train, y_train)
# Make predictions and evaluate
y_pred = model.predict(X_test)
_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error with 4 features: {_rmse}')
predictions = pd.DataFrame({f'Actual {dependent}': y_test, f'Predicted {dependent}': y_pred})
print(predictions.head())

