import numpy as np
from config import AirDensity

# Function to calculate relative velocity
def calculate_relative_velocity(v, w, wind_dir, car_dir=135):
    theta = np.radians(wind_dir - car_dir)  # Convert to radians and find the angle difference
    return np.sqrt(v**2 + w**2 + 2 * v * w * np.cos(theta))

# Funciton to calcualte drag force
def get_drag_force(CDA, speed, wind_speed, dir, wind_dir):
    speed = calculate_relative_velocity(speed, wind_speed, wind_dir, dir)
    return (0.5 * CDA * AirDensity * speed**2)
