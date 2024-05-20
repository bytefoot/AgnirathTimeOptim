import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt

import config
from solar import Solar
from constraints import get_bounds, constraint_battery, objective, constraint_acceleration, constraint_battery2
from profiles import extract_profiles

EPSILON = 1e-8

class Motor:
    def __init__(self, wheel_radius, mass, wheels, CDA, zero_speed_crr):
        self.wheel_radius = wheel_radius  # inches
        self.mass = mass  # kg
        self.CDA = CDA
        self.zero_speed_crr = zero_speed_crr  # 0.003
        self.no_of_wheels = wheels

    # def calculate_power(self, speed, acceleration, slope, dir, wind_speed, wind_dir):
    def calculate_power(self, speed, acceleration, slope):

        # Calculate power required to overcome rolling resistance and aerodynamic drag
        self.dynamic_speed_crr = (self.no_of_wheels / 3) * 4.1 * 10 ** (-5) * speed
        
        rolling_resistance = (self.mass * 9.8 * (self.zero_speed_crr + self.dynamic_speed_crr))  # Assume coefficient of friction = 0.01
        
        # drag_force = get_drag_force(self.CDA, speed, wind_speed, wind_dir, dir)
        # drag_force = 0.5 * self.CDA * config.AirDensity * (speed ** 2)
        drag_force = 0.5 * self.CDA * config.AirDensity * ((speed)*2+(12.57)* 2-2*(speed)*(12.57)*np.cos(np.radians(10)))
        
        power = (
            rolling_resistance + drag_force
            + self.mass * acceleration
            + self.mass * config.g * np.sin(slope)
        ) * abs(speed)

        return max(power, 0)


class ElectricCar:
    def __init__(self, motor):
        # Motor
        self.motor = motor

    def drive_sim(
        self,
        start_speed, stop_speed, dx, slope
    ):
        speed = (start_speed + stop_speed) / 2

        # instantaneous time elapsed 
        dt = self.calculate_dt(start_speed, stop_speed, dx)

        # current power consumption
        power = self.motor.calculate_power(
            speed, (stop_speed - start_speed) / dt,
            slope
        )
        # Current energy consumption
        energy_consumed = power * dt / 3600.0

        return (
            dt,
            dx,
            power,
            energy_consumed
        )
    
    def calculate_dt(self, start_speed, stop_speed, dx):
        dt = 2 * dx /(start_speed + stop_speed + EPSILON)
        # print(dt)
        return dt


def main():
    route_df = pd.read_csv("./data/raw/temp_route_data.csv")

    motor = Motor(
        config.WheelRadius, config.Mass, config.Wheels,
        config.CDA, config.ZeroSpeedCrr
    )
    car = ElectricCar(motor)
    solar_panel = Solar(config.PanelEfficiency, config.PanelArea)

    N_V = len(route_df) + 1
    velocity_profile = np.ones(N_V) * config.InitialGuessVelocity

    bounds = get_bounds(N_V)
    constraints= [
        {
            "type": "ineq",
            "fun": constraint_battery,
            "args": (
                car, solar_panel, route_df,
                config.BatteryCapacity * (1 - config.DeepDischargeCap),
            )
        },
        {
            "type": "ineq",
            "fun": constraint_battery2,
            "args": (
                car, solar_panel, route_df,
                config.BatteryCapacity * (1 - config.DeepDischargeCap)
            )
        },
        {
            "type": "ineq",
            "fun": constraint_acceleration,
            "args": (
                car, route_df
            )
        }
    ]

    print("Starting Optimisation")
    print("="*60)
    optimised_velocity_profile = optimize.minimize(
        objective, velocity_profile,
        args=(car, route_df),
        bounds=bounds,
        method=config.ModelMethod,
        constraints=constraints,
        options={
            'verbose': 3,
        }
    )
    optimised_velocity_profile = optimised_velocity_profile.x

    print("done.")
    print("Total time taken for race:", objective(np.array(optimised_velocity_profile), car, route_df), "\bs")
    # print(optimised_velocity_profile)

    # route_df['CumulativeDistance(km)'], optimised_velocity_profile
    outdf = pd.DataFrame(
        dict(zip(
            ['CummulativeDistance', 'Velocity', 'Acceleration', 'Battery', 'EnergyConsumption', 'Solar', 'Time'],
            extract_profiles(car, solar_panel, optimised_velocity_profile, route_df)
        ))
    )

    outdf.to_csv("./runlog/run_dat.csv", index=False)
    print("Written results to `run_dat.csv`")

if __name__ == "__main__":
    main()
