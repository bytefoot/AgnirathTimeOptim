import numpy as np

import numpy as np
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import math

import config
from solar import Solar
from constraints import get_bounds, constraint_battery, objective, constraint_acceleration, constraint_battery2
from profiles import extract_profiles

EPSILON = 1e-8

def efficiency(omega):
    import math
    m = 270  # mass of car
    u1 = 0.0045 # static friction coefficient
    rho = 1.192 # air density
    a = 0.92 # frontal area of car
    Cd = 0.098 # coefficient of drag
    r_in = 0.214 # inner radius of wheel
    r_out = 0.2785  # outer radius of wheel
    Ta = 295
    visc = 1.524 * 10**-5  # kinematic viscosity of air
    v_rotor = omega * r_in  # circumferential speed of rotor
    v_car  = omega*r_out  
    RN = v_rotor * r_out / visc  # Reynolds number
    g = 1.5 * 10**-3  # air gap spacing between stator and rotor
  
    
    pi = math.pi
    
    if RN > 0.8 * 10**5:
        # Regime III
        Cf = 0.08 / (((g / r_out) ** 0.167) * (RN ** 0.25))
    else:
        # Regime I
        Cf = 2 * pi * r_out / (g * RN)
    
    t = r_out * ((m * 9.81 * u1) + (0.5 * Cd * a * rho * (omega ** 2) * (r_out ** 2))) + 0.5 * Cf * rho * pi * (omega ** 2) * ((r_out ** 5) - (r_in ** 5))
    
    def find_winding_temp(Tw):
        B = 1.32 - 1.2 * 10**-3 * (Ta / 2 + Tw / 2 - 293)  # magnetic remanence
        i = 0.561 * B * t  # RMS phase current
        R = 0.0575 * (1 + 0.0039 * (Tw - 293))  # resistance of windings
        Pc = 3 * i ** 2 * R  # copper (ohmic) losses
        Pe = 9.602 * 10**-6 * (B * omega) ** 2 / R  # eddy current losses
        Tw2 = 0.455 * (Pc + Pe) + Ta

        if np.abs(Tw2 - Tw) < 0.001:
            return Tw2
        else:
            return find_winding_temp(Tw2)  # iterating till difference becomes less than 0.001
    
    Tw = find_winding_temp(Ta)
    B = 1.32 - 1.2 * 10**-3 * (Ta / 2 + Tw / 2 - 293)
    i = 0.561 * B * t
    R = 0.0575 * (1 + 0.0039 * (Tw - 293))
    Pc = 3 * i ** 2 * R
    Pe = 9.602 * 10**-6 * (B * omega) ** 2 / R
    Pw = (omega ** 2) * (170.4 * 10**-6)  # windage losses
    t_f = 0.5 * Cf * rho * pi * (omega ** 2) * ((r_out ** 5) - (r_in ** 5))  # frictional torque due to dynamic rolling resistance
    Pf = t_f * omega  # wheel losses (frictional losses)
    P_out = t * omega  # output power
    P_in = P_out + Pc + Pe + Pw  # input power
    P_loss = Pc + Pe + Pw  # total power loss
    eta = P_out * 100 / P_in  # efficiency of motor
    return eta
class Motor:
    def __init__(self, wheel_radius, mass, wheels, CDA, zero_speed_crr):
        self.wheel_radius = wheel_radius  
        self.mass = mass  # kg
        self.CDA = CDA
        self.zero_speed_crr = zero_speed_crr  
        self.no_of_wheels = wheels

        # Precompute constants
        self.rolling_resistance = self.mass * 9.81 * 0.0045  # Assume coefficient of friction = 0.01
        self.wind_speed_factor = 12.57*5/18
        self.wind_dir_factor = np.cos(np.radians(10))

    def calculate_power(self, speed, acceleration, slope):
        # Calculate omega (angular velocity)
        omega = speed / self.wheel_radius

        # Calculate efficiency
        eta = efficiency(omega) / 100  # Convert efficiency to fraction

        # Calculate power required to overcome rolling resistance and aerodynamic drag
        drag_force = 0.5 * self.CDA * config.AirDensity * (speed ** 2 + self.wind_speed_factor ** 2 - 2 * speed * self.wind_speed_factor * self.wind_dir_factor)

        power = (
            self.rolling_resistance + drag_force
            + self.mass * acceleration   # Adjusted for efficiency
            + self.mass * config.g * np.sin(slope)
        ) * abs(speed)/eta
        # print(drag_force*speed,self.mass * acceleration*speed, self.rolling_resistance * speed)
        return max(power, 0)

class ElectricCar:
    def __init__(self, motor):
        # Motor
        self.motor = motor

    def drive_sim(self, start_speed, stop_speed, dx, slope):
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

        return dt, dx, power, energy_consumed
    
    def calculate_dt(self, start_speed, stop_speed, dx):
        dt = 2 * dx / (start_speed + stop_speed + EPSILON)
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
    constraints = [
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
    print("=" * 60)
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
    print("Total time taken for race:", objective(np.array(optimised_velocity_profile), car, route_df), "s")

    outdf = pd.DataFrame(
        dict(zip(
            ['CummulativeDistance', 'Velocity', 'Acceleration', 'Battery', 'EnergyConsumption', 'Solar', 'Time'],
            extract_profiles(car, solar_panel, optimised_velocity_profile, route_df)
        ))
    )

    outdf.to_csv("run_dat.csv", index=False)
    print("Written results to `run_dat.csv`")

if __name__ == "__main__":
    main()
