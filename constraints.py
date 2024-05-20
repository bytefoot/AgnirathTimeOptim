import config
import numpy as np

# Bounds for the velocity
def get_bounds(N):
    return ([(0, 0)] + [(0.01, config.MaxVelocity)]*(N-2) + [(0, 0)])

def objective(velocity_profile, car, route_df, sampling_rate):
    cummulative_time = 0
    N_V = len(velocity_profile)

    for i in range(N_V-1):
        approx_a = (velocity_profile[i+1] - velocity_profile[i])/sampling_rate
        for j in range(0, sampling_rate):
            cummulative_time += car.calculate_dt(
                velocity_profile[i] + approx_a * j, velocity_profile[i] + approx_a * (j+1),
                route_df.iloc[i*sampling_rate+j, 0]
            )

    return cummulative_time


def constraint_battery(v_prof, car, solar_panel, route_df, safe_battery_capacity, sampling_rate):
    battery_level = safe_battery_capacity
    min_bl = battery_level
    max_bl = 0
    time_elapsed = 0

    for i in range(len(v_prof)-1):
        approx_a = (v_prof[i+1] - v_prof[i])/sampling_rate
        for j in range(0, sampling_rate):
            dt, dx, P, dE = car.drive_sim(
                v_prof[i] + approx_a * j, v_prof[i+1] + approx_a * (j+1),
                route_df.iloc[i*sampling_rate+j, 0], route_df.iloc[i*sampling_rate+j, 2]
            )

            solE = solar_panel.calculate_energy(dt, time_elapsed, route_df.iloc[i*sampling_rate+j, 3], route_df.iloc[i*sampling_rate+j, 4])
            battery_level +=  - dE + solE

            min_bl = min(battery_level, min_bl)
            max_bl = max(battery_level, max_bl)

            time_elapsed += dt

    return min_bl

def constraint_battery2(v_prof, car, solar_panel, route_df, safe_battery_capacity, sampling_rate):
    battery_level = safe_battery_capacity
    min_bl = battery_level
    max_bl = 0
    time_elapsed = 0

    for i in range(len(v_prof)-1):
        approx_a = (v_prof[i+1] - v_prof[i])/sampling_rate
        for j in range(0, sampling_rate):
            dt, dx, P, dE = car.drive_sim(
                v_prof[i] + approx_a * j, v_prof[i+1] + approx_a * (j+1),
                route_df.iloc[i*sampling_rate+j, 0], route_df.iloc[i*sampling_rate+j, 2]
            )

            solE = solar_panel.calculate_energy(dt, time_elapsed, route_df.iloc[i*sampling_rate+j, 3], route_df.iloc[i*sampling_rate+j, 4])
            battery_level +=  - dE + solE

            min_bl = min(battery_level, min_bl)
            max_bl = max(battery_level, max_bl)

            time_elapsed += dt

    # print(min_bl, (safe_battery_capacity - max_bl))
    return (safe_battery_capacity - max_bl)

def constraint_acceleration(v_prof, car, route_df, sampling_rate):
    max_accel = 0
    for i in range(len(v_prof)-1):
        approx_a = (v_prof[i+1] - v_prof[i])/sampling_rate
        for j in range(0, sampling_rate):
            dt = car.calculate_dt(
                v_prof[i] + approx_a * j, v_prof[i] + approx_a * (j+1),
                route_df.iloc[i*sampling_rate+j, 0]
            )
    
            max_accel = max(max_accel, approx_a / dt)
    
    return config.MaxAcceleration - max_accel