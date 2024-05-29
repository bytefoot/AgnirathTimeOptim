import config
import numpy as np
import math
def efficiency(omega):
    m = 270  # mass of car
    u1 = 0.0045 # static friction coefficient
    rho = 1.192 # air density
    a = 1  # frontal area of car
    Cd = 0.092  # coefficient of drag
    r_in = 0.214 # inner radius of wheel
    r_out = 0.2785  # outer radius of wheel
    Ta = 295
    visc = 1.524 * 10**-5  # kinematic viscosity of air
    v_rotor = omega * r_in  # circumferential speed of rotor
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
    return P_out

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


def constraint_battery(v_prof, car, solar_panel, route_df, safe_battery_capacity):
    battery_level = safe_battery_capacity
    min_bl = battery_level
    max_bl = 0
    time_elapsed = 0

    for i in range(len(v_prof)-1):
        dt, dx, P, dE = car.drive_sim(
            v_prof[i], v_prof[i+1],
            route_df.iloc[i, 0], route_df.iloc[i, 2]
        )

        solE = solar_panel.calculate_energy(dt, time_elapsed, route_df.iloc[0, 3], route_df.iloc[0, 4])
        battery_level +=  - dE + solE

        min_bl = min(battery_level, min_bl)
        max_bl = max(battery_level, max_bl)

        time_elapsed += dt

    return min_bl

def constraint_battery2(v_prof, car, solar_panel, route_df, safe_battery_capacity):
    battery_level = safe_battery_capacity
    min_bl = safe_battery_capacity
    max_bl = 0
    time_elapsed = 0

    for i in range(len(v_prof)-1):
        dt, dx, P, dE = car.drive_sim(
            v_prof[i], v_prof[i+1],
            route_df.iloc[i, 0], route_df.iloc[i, 2]
        )

        solE = solar_panel.calculate_energy(dt, time_elapsed, route_df.iloc[0, 3], route_df.iloc[0, 4])
        battery_level +=  - dE + solE

        min_bl = min(battery_level, min_bl)
        max_bl = max(battery_level, max_bl)

        time_elapsed += dt

    # print(min_bl, (safe_battery_capacity - max_bl))
    return (safe_battery_capacity - max_bl)
def constraint_acceleration(v_prof, car, route_df):
    max_accel = 2
    for i in range(len(v_prof) - 1):
        dt = car.calculate_dt(v_prof[i], v_prof[i + 1], route_df.iloc[i, 0])
        omega = v_prof[i] / car.motor.wheel_radius
        P_out = efficiency(omega)
        accel = P_out / (car.motor.mass * v_prof[i])
        max_accel = max(max_accel, accel)
    
    return config.MaxAcceleration - max_accel
