import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the app
st.title("Data Analysis Project: Bike Sharing Dataset")

# Input user information
st.markdown("""
- **Nama:** Ahmad Alfin Nur Hakim  
- **Email:** ahmadalfin130804@gmail.com  
- **ID Dicoding:** ahmad_alf
""")

# Load data directly from the specified path
st.header("Load Data")
bike_day = pd.read_csv("day.csv")
bike_hour = pd.read_csv("hour.csv")

# Display the first few rows of the dataset
st.write(bike_day.head())

# Assessing Data
st.subheader("Assessing Data")
st.write(bike_day.info())
st.write(bike_day.describe(include="all"))

# Cleaning Data
bike_day.dropna(axis=0, inplace=True)
bike_day['dteday'] = pd.to_datetime(bike_day['dteday'])
bike_day['weekday'] = bike_day['dteday'].dt.day_name()

st.write("Data cleaned and processed.")

# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis (EDA)")

# RFM Calculation
current_date = bike_day['dteday'].max()
rfm = bike_day.groupby('casual').agg({
    'dteday': lambda x: (current_date - x.max()).days,  # Recency
    'cnt': 'count'  # Frequency
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency']

# Visualization 1: Frequency of Rentals
st.subheader("Distribution of Recency by Frequency of Bicycle Rentals")
plt.figure(figsize=(12, 6))
sns.boxplot(data=rfm, x='Frequency', y='Recency', palette='husl')
plt.title('Distribution of Recency by Frequency of Bicycle Rentals')
plt.xlabel('Frequency of Rentals')
plt.ylabel('Recency (days since last rental)')
plt.grid(axis='y')
st.pyplot(plt)

# Calculate total borrowing by day of the week
bike_day['total_bike'] = bike_day['casual'] + bike_day['registered']
usage_by_weekday = bike_day.groupby('weekday')['total_bike'].sum().reset_index()
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
usage_by_weekday['weekday'] = pd.Categorical(usage_by_weekday['weekday'], categories=ordered_days, ordered=True)
usage_by_weekday = usage_by_weekday.sort_values('weekday')

# Visualization 2: Total Bike Usage by Day of the Week
st.subheader("Total Bike Usage by Day of the Week")
plt.figure(figsize=(12, 6))
sns.barplot(x='total_bike', y='weekday', data=usage_by_weekday, palette='Set2')
plt.title('Total Bike Usage by Day of the Week')
plt.xlabel('Total Bicycle Usage (in units of 1)')
plt.ylabel('Days of the Week')
plt.grid(axis='x')
st.pyplot(plt)

# Weather impact analysis
peak_hours = bike_hour[(bike_hour['hr'] >= 7) & (bike_hour['hr'] <= 10)]
weather_impact = peak_hours.groupby('weathersit')['cnt'].sum().reset_index()

# Visualization 3: Weather Impact on Bike Rentals
st.subheader("Weather Impact on Bike Rentals During Peak Hours (07:00 - 10:00)")
plt.figure(figsize=(10, 6))
plt.barh(weather_impact['weathersit'], weather_impact['cnt'], color='skyblue')
plt.ylabel('Weather Conditions')
plt.xlabel('Total Bike Loans')
plt.title('Weather Impact on Bike Rentals')
plt.grid(axis='x')
plt.gca().invert_yaxis()
st.pyplot(plt)

# Conclusion
st.header("Conclusion")
st.write("""
- Most active customers only use one loan in one day since the last loan.
- The use of bicycles indicates that they are utilized for daily needs.
- Increase the number of bikes available during sunny weather conditions as it attracts more customers.
""")