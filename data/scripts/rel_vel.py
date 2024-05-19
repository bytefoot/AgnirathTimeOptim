import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('slope_profile2.csv')

# Wind data for cities (manually created dictionary from the provided data)
wind_data = {
    'CumulativeDistance(km)': [0, 400, 800, 1200, 1600, 2000, 2400, 2800],
    'WindSpeed(m/s)': [12.15, 15.33, 8.8, 18.5426, 9.65, 15.716, 15.835, 10.72],
    'WindDirection(deg)': [90, 112.5, 112.5, 112.5, 90, 45, 45, 0] # example wind directions in degrees
}

wind_df = pd.DataFrame(wind_data)

# Interpolate wind speed and direction
df['WindSpeed(m/s)'] = np.interp(df['CumulativeDistance(km)'], wind_df['CumulativeDistance(km)'], wind_df['WindSpeed(m/s)'])
df['WindDirection(deg)'] = np.interp(df['CumulativeDistance(km)'], wind_df['CumulativeDistance(km)'], wind_df['WindDirection(deg)'])

# Car's travel direction (southeast), angle with respect to north is 135 degrees
car_direction_deg = 135

# Function to calculate relative velocity
def calculate_relative_velocity(v, w, wind_dir, car_dir=car_direction_deg):
    theta = np.radians(wind_dir - car_dir)  # Convert to radians and find the angle difference
    relative_velocity = np.sqrt(v**2 + w**2 + 2 * v * w * np.cos(theta))
    return relative_velocity

# Assume a constant car speed 'v'
car_speed = 25  # example car speed in m/s

# Calculate relative velocity for each row
df['RelativeVelocity(m/s)'] = df.apply(lambda row: calculate_relative_velocity(car_speed, row['WindSpeed(m/s)'], row['WindDirection(deg)']), axis=1)


df.to_csv('slope_profile_with_relative_velocity.csv', index=False)

print(df.head())
