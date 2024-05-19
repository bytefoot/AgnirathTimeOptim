import numpy as np
from scipy.optimize import minimize

# Constants
MAX_ACCELERATION = 3.0  
MAX_DECELERATION = 3.0  
V_MAX = 36.0  # m/s
SAFE_PERCENTAGE = 0.2  
BATTERY_CAPACITY = 100.0  

# Define configuration parameters
class Config:
    AirDensity = 1.225  
    g = 9.81  

config = Config()



class Motor:
    def calculate_power(self, speed, acceleration, slope):
        # Calculate power required to overcome rolling resistance and aerodynamic drag
        dynamic_speed_crr = (4 / 3) * 4.1 * 10 ** (-5) * speed  # assuming no_of_wheels = 4
        
        rolling_resistance = (1500 * 9.8 * (0.01 + dynamic_speed_crr))  # mass = 1500kg
        
        drag_force = 0.5 * 0.5 * config.AirDensity * (speed ** 2)  # CDA = 0.5
        
        power = (
            rolling_resistance + drag_force
            + 1500 * acceleration
            + 1500 * config.g * np.sin(slope)
        ) * abs(speed)
        
        return power

class Car:
    def __init__(self, mass):
        self.mass = mass
        self.motor = Motor()

    def calculate_dt(self, start_speed, stop_speed, dx):
        return 2 * dx / (start_speed + stop_speed)

    def drive_sim(self, start_speed, stop_speed, dx, battery_capacity, slope):
        dt = self.calculate_dt(start_speed, stop_speed, dx)
        acceleration = (stop_speed - start_speed) / dt
        speed = (start_speed + stop_speed) / 2
        
        power = self.motor.calculate_power(speed, acceleration, slope)
        energy_consumed = power * dt / 3600

        return dt, dx, power, energy_consumed

# Define Solar class
class Solar:
    def __init__(self, efficiency, area):
        self.efficiency = efficiency
        self.area = area

    def calculate_energy(self, power, time, solar_irradiance, angle):
        return self.efficiency * self.area * solar_irradiance * np.cos(np.radians(angle)) * time

# Define objective function
def objective(x, *args):
    car, x_in, route_df, battery_capacity = args
    E_use = 0
    total_time = 0
    n_v = len(x_in)
    solar_panel = Solar(0.17, 4)
    
    for i in range(n_v-1):
        sim1 = car.drive_sim(x[i], x[i+1], x_in[i+1] - x_in[i], battery_capacity - E_use, route_df[i, 1])
        E_use += sim1[3]
        E_use -= solar_panel.calculate_energy(sim1[2], sim1[0], route_df[i, 2], route_df[i, 3]) / 3600
        total_time += sim1[0]
    
    sim_last = car.drive_sim(x_in[-1], 0, 0, battery_capacity - E_use, route_df[-1, 1])
    E_use += sim_last[3]
    E_use -= solar_panel.calculate_energy(sim_last[2], sim_last[0], route_df[-1, 2], route_df[-1, 3]) / 3600
    total_time += sim_last[0]

    return total_time

# Define constraints
def constraint_velocity(x):
    return V_MAX - x

def constraint_acceleration(x, car, dx):
    accels = [(x[i+1] - x[i]) / car.calculate_dt(x[i], x[i+1], dx) for i in range(len(x)-1)]
    return [MAX_ACCELERATION - a for a in accels]

def constraint_deceleration(x, car, dx):
    decels = [(x[i] - x[i+1]) / car.calculate_dt(x[i], x[i+1], dx) for i in range(len(x)-1)]
    return [MAX_DECELERATION + d for d in decels]

def constraint_battery(x, car, dx, route_df, battery_capacity):
    E_use = 0
    for i in range(len(x)-1):
        sim1 = car.drive_sim(x[i], x[i+1], dx, battery_capacity - E_use, route_df[i, 1])
        E_use += sim1[3]
    E_use += car.drive_sim(x[-1], 0, 0, battery_capacity - E_use, route_df[-1, 1])[3]
    return (battery_capacity - E_use) - battery_capacity * SAFE_PERCENTAGE

# Setup the problem
car = Car(1500)  # mass in kg
x_in = [0, 10, 20, 30, 40, 50]  # example positions in meters
route_df = np.array([[0, 0, 800, 30], [10, 0.01, 850, 30], [20, 0.01, 900, 30], [30, 0.02, 950, 30], [40, 0.02, 1000, 30], [50, 0, 1050, 30]])  # example route data

# Initial guess for velocities (linear increase)
initial_guess = np.linspace(0, V_MAX, len(x_in))

# Constraints
cons = [
    {'type': 'ineq', 'fun': constraint_velocity},
    {'type': 'ineq', 'fun': constraint_acceleration, 'args': (car, 10)},
    {'type': 'ineq', 'fun': constraint_deceleration, 'args': (car, 10)},
    {'type': 'ineq', 'fun': constraint_battery, 'args': (car, 10, route_df, BATTERY_CAPACITY)}
]

# Perform optimization
result = minimize(objective, initial_guess, args=(car, x_in, route_df, BATTERY_CAPACITY), constraints=cons, method='trust-constr', options={'disp': True})

# Result
if result.success:
    print("Optimal solution found:", result.x)
    print("Minimum time:", result.fun)
else:
    print("Optimization failed:", result.message)
