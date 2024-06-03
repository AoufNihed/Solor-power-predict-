import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv(r'C:\Users\hey\OneDrive\Bureau\solor power\Solar.csv')

# Data preprocessing
df['Dates'] = pd.to_datetime(df['Date-Hour(NMT)'], dayfirst=True).dt.date
df['Hour'] = pd.to_datetime(df['Date-Hour(NMT)'], dayfirst=True).dt.hour
df['Day'] = pd.to_datetime(df['Date-Hour(NMT)'], dayfirst=True).dt.day
df['Month'] = pd.to_datetime(df['Date-Hour(NMT)'], dayfirst=True).dt.month
df['Year'] = pd.to_datetime(df['Date-Hour(NMT)'], dayfirst=True).dt.year

df = df.drop(['Date-Hour(NMT)'], axis=1)

# Encode categorical variables
df['Season'] = df['Month'].apply(lambda x: "Winter" if x < 3 or x > 11 else "Spring" if 3 <= x < 6 else "Summer" if 6 <= x < 9 else "Autumn")
df['Time'] = df.apply(lambda row: 'Day' if (7 < row['Hour'] < 18) and (row['Season'] == 'Winter') else 'Night', axis=1)

# Convert categorical columns to numeric
df = pd.get_dummies(df, columns=['Season', 'Time'], drop_first=True)

# Ensure all columns are numeric except 'Dates'
numeric_df = df.select_dtypes(include=[np.number])

# Split data
X = numeric_df.drop(['SystemProduction'], axis=1)
y = numeric_df['SystemProduction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
st.title('Solar Power Generation Analysis 🌞')
st.write('This app provides insights and visualizations for solar power generation data.')

if st.button('Load Data'):
    st.write(df.head())

if st.button('Visualize Correlations'):
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()

st.write('The following parameters influence solar power generation:')
st.write('Temperature, Humidity, Wind Speed, Radiation, Air Pressure')

if st.button('Visualize Data'):
    sns.pairplot(numeric_df, diag_kind='kde')
    st.pyplot()

st.write('Random Forest Model Performance:')
st.write(f'Mean Squared Error: {mse}')

# Define models
st.write('The models used in this analysis include:')
st.write('- Random Forest Regressor')

# Display model summary
if st.button('Show Model Summary'):
    st.write(f'Model: Random Forest Regressor\nMean Squared Error: {mse}')
