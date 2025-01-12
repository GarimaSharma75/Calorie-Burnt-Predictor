
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as regressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from PIL import Image
import time

st.title(' Calorie Burnt Predictor')

# Load the CSV files into DataFrames
df1 = pd.read_csv("calories.csv")
df2 = pd.read_csv("exercise.csv")

#Heading
st.title('Calorie Burnt Predictor')

image = Image.open("burn.png")
st.image(image, caption="Calorie Predictor", use_column_width=True)

# Take inputs
user_input_age = st.sidebar.slider('Age', 1, int(df2["Age"].max()), 18)
user_input_height = st.sidebar.slider('Height', 1, int(df2["Height"].max()), 170)
user_input_weight = st.sidebar.slider('Weight', 1, int(df2["Weight"].max()), 60)
user_input_duration = st.sidebar.slider('Duration', 1, int(df2["Duration"].max()), 12)
user_input_rate = st.sidebar.slider('Heart Rate', 1, int(df2["Heart_Rate"].max()), 80)
user_input_temp = st.sidebar.slider('Body Temperature', 1, int(df2["Body_Temp"].max()), 39)

# Split data
X = df2.drop(['User_ID', 'Gender'], axis=1)  # Features (dropping User_ID and Gender)
Y = df1['Calories']  # Target (Calories)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scale the data
scaler = StandardScaler()  # Instantiate StandardScaler
X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test = scaler.transform(X_test)  # Transform the test data

# Create the model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Create a DataFrame for the user input
user_input_cal = pd.DataFrame({
    'Age': [user_input_age],
    'Height': [user_input_height],
    'Weight': [user_input_weight],
    'Duration': [user_input_duration],
    'Heart Rate': [user_input_rate],
    'Body Temperature': [user_input_temp],
}, columns=X.columns)

# Scale the user input
user_input_scaled = scaler.transform(user_input_cal)  # Transform user input with the fitted scaler

# Display a loading screen
with st.spinner('Calculating...'):
    time.sleep(2)

# Predict using the model
predicated_calories = model.predict(user_input_scaled)  # Use the trained XGBRegressor model

# Print the predicted median calories
st.write(f'The predicted calories value for given input is: {predicated_calories[0]}')

