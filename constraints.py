import config
import numpy as np

# Bounds for the velocity
def get_bounds(N):
    return ([(0, 0)] + [(0.01, config.MaxVelocity)]*(N-2) + [(0, 0)])

def objective(velocity_profile, car, route_df):
    cummulative_time = 0
    N_V = len(velocity_profile)

    for i in range(N_V-1):
        dt = car.calculate_dt(
            velocity_profile[i], velocity_profile[i+1],
            route_df.iloc[i, 0]
        )

        cummulative_time += dt

    # print(cummulative_time)
    return cummulative_time


def constraint_battery(v_prof, car, solar_panel, route_df, safe_battery_capacity, start_time):
    max_energy_utilisation = 0
    cumE = 0
    time_elapsed = 0

    for i in range(len(v_prof)-1):
        dt, dx, P, dE = car.drive_sim(
            v_prof[i], v_prof[i+1],
            route_df.iloc[i, 0], route_df.iloc[i, 2]
        )

        dE -= solar_panel.calculate_energy(dt, start_time+time_elapsed, route_df.iloc[0, 3], route_df.iloc[0, 4])
        cumE += dE
        max_energy_utilisation = max(cumE, max_energy_utilisation)

        time_elapsed += dt
    
    return safe_battery_capacity - max_energy_utilisation

def constraint_acceleration(v_prof, car, route_df):
    max_accel = 0
    for i in range(len(v_prof)-1):
        dt = car.calculate_dt(v_prof[i], v_prof[i+1], route_df.iloc[i, 0])
        max_accel = max(max_accel, (v_prof[i+1] - v_prof[i]) / dt)
    
    return config.MaxAcceleration - max_accel