import numpy as np
import config

def extract_profiles(car, solar_panel, velocity_profile, route_df):

    # Variables to keep track of:
    time_elapsed = 0
    time = [0]
    acceleration = []
    energy_consumpution = []
    solar_energy = []
    distance_travelled = 0

    for i in range(len(velocity_profile)-1):
        dt, dx, P, dE = car.drive_sim(
            velocity_profile[i], velocity_profile[i+1],
            route_df.iloc[i, 0], route_df.iloc[i, 2]
        )

        solE = solar_panel.calculate_energy(dt, time_elapsed, route_df.iloc[0, 3], route_df.iloc[0, 4])

        energy_consumpution.append(dE)
        solar_energy.append(solE)
        acceleration.append((velocity_profile[i+1] - velocity_profile[i])/dt)

        time_elapsed += dt
        distance_travelled += dx
        time.append(time_elapsed)
    
    velocity_profile, acceleration_profile, energy_consumpution_profile, solar_energy_profile = map(np.array, (velocity_profile, acceleration, energy_consumpution, solar_energy))
    
    net_energy_profile = energy_consumpution_profile.cumsum() - solar_energy_profile.cumsum()
    battery_profile = config.BatteryCapacity - net_energy_profile
    battery_profile = np.concatenate((np.array([config.BatteryCapacity]), battery_profile))

    battery_profile = battery_profile * 100 / config.BatteryCapacity
    energy_consumpution=energy_consumpution_profile.cumsum()
    distances = np.append([0], route_df['CumulativeDistance(km)'])

    return [
        distances,
        velocity_profile,
        np.concatenate((np.array([np.nan]), acceleration_profile,)),
        battery_profile,
        np.concatenate((np.array([np.nan]), energy_consumpution,)),
        np.concatenate((np.array([np.nan]), solar_energy_profile)),
        np.array(time)
    ]