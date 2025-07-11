import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing          #load_boston was removed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st
from sklearn.preprocessing import StandardScaler

housing_data = fetch_california_housing()
df = pd.DataFrame(data = housing_data.data, columns = housing_data.feature_names)

st.title("California Housing Dataset")

x = housing_data.data
y = housing_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

MedInc = st.slider("Enter the Median Income of households within the block ($10000s): ", 0.5, 15.0, 0.5)
HouseAge = st.slider("Enter the House Age (years): ", 1, 52, 1)
AveRooms = st.slider("Enter the Average Number of Rooms within the block: ", 1, 10, 1)
AveBedrms = st.slider("Enter the Average Number of Bedrooms within the block: ", 1, 35, 1)
Population = st.slider("Enter the Population of the block: ", 1, 36000, 1)
AveOccup = st.slider("Enter the Average Number of Occupants: ", 1, 100, 1)
Latitude = st.slider("Enter the Latitude: ", 32.0, 42.0, 32.0)
Longitude = st.slider("Enter the Longitude: ", -124.0, -114.0, -124.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
input_data = scaler.transform(input_data)
price_pred = model.predict(input_data)
st.subheader("Predicted House Price: $ " + str(price_pred[0]*100000))

evaluate = st.checkbox("Evaluate Model")
if evaluate:
   st.subheader("Model Evaluation")
   st.write("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
   st.write("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
   st.write("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
   st.write("R2 Score: ", r2_score(y_test, y_pred))

stats = st.checkbox("Show Data Statistics")
if stats:
   st.subheader("Data Statistics")
   st.write("Descriptive Statistics : " ,df.describe())
   st.write("Number of Null Values in each column : ",df.isnull().sum())
            