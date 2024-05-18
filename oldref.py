
# In this model I attempt to implement cache for API Calls. This model is fully functional

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import math 
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import random
import solcast
from solcast import forecast
import concurrent.futures

route = [[10000, 1*0.015708]
    , [2000, 4*0.015708]
    , [5000, 2*0.015708]
    , [3000, 1*0.015708],[7000, 2*0.015708],[10000, 3*0.015708]
    ]
cache = {}


class Motor:
    def __init__(
        self, wheel_radius, mass, wheels, aerodynamic_coef, frontal_area, zero_speed_crr
    ):
        self.wheel_radius = wheel_radius  # inches
        self.mass = mass  # kg
        self.aerodynamic_coef = aerodynamic_coef
        self.frontal_area = frontal_area  # m^2
        self.zero_speed_crr = zero_speed_crr  # 0.003
        self.no_of_wheels = wheels

    def calculate_power(self, speed, acceleration, slope):
        # Calculate power required to overcome rolling resistance and aerodynamic drag
        self.dynamic_speed_crr = (self.no_of_wheels / 3) * 4.1 * 10 ** (-5) * speed
        rolling_resistance = (self.mass * 9.8 * (self.zero_speed_crr + self.dynamic_speed_crr))  # Assume coefficient of friction = 0.01
        drag_force = (self.frontal_area * 0.5 * self.aerodynamic_coef * 1.225 * speed**2)  # Air density = 1.225 kg/m^3
        power = (rolling_resistance + drag_force + self.mass*acceleration + self.mass*9.8*np.sin(slope)) * abs(speed)
        return power



class ElectricCar:
    def __init__(self, motor, distance, battery_capacity, route):
        self.motor = motor  
        self.dt = 1  # seconds
        self.start_speed = 0  # m/s

        self.route = route  # [distance, elevation]
        self.distance = distance  # meters

        # Battery
        self.remaining_energy = battery_capacity

    def drive_sim(
        self, start_speed, stop_speed, acceleration, remaining_energy, distance_to_travel, slope
    ):  
        self.acceleration = acceleration  # m/s^2
        self.speed = start_speed
        remaining_energy = remaining_energy
        self.distance_to_travel = distance_to_travel
        self.slope = slope

        self.energy_consumed_car = 0  # Wh
        self.distance_traveled = 0  # m/s
        self.time_elapsed = 0  # seconds


        ctr = 0
        
        self.time_elapsed = 2*distance_to_travel/(self.speed+stop_speed)
        # print(self.time_elapsed)
        self.power = self.motor.calculate_power(self.speed, (stop_speed - start_speed)/self.time_elapsed, self.slope)
        self.energy_consumed_car = self.power*self.time_elapsed/3600
        remaining_energy -= self.energy_consumed_car    
        # while self.distance_traveled < distance_to_travel and self.remaining_energy > 1:  
        #     # ctr +=1 
        #     # if (ctr == 10000):
        #         # print(self.speed)  
        #     if self.speed < 30:
        #         self.speed += self.acceleration * self.dt
        #     elif self.speed >= 30:
        #         self.speed = 30
        #         acceleration = 0
        #     if self.speed <= stop_speed:
        #         self.speed = stop_speed
        #         acceleration = 0

        #     # print(self.speed)
            
        #     # print("hi")
        #     self.power = self.motor.calculate_power(self.speed,self.acceleration,self.slope)
        #     self.time_elapsed += self.dt
        #     self.energy = self.power * self.dt / 3600
        #     self.instantaneous_distance = self.speed * self.dt
        #     self.energy_consumed_car += self.energy
        #     remaining_energy -= self.energy
        #     self.distance_traveled += self.instantaneous_distance
        # print(
        #     f"Time: {self.time_elapsed} seconds, Distance: {self.distance_traveled:.2f} meters, Speed:{self.speed:.3f} m/s, Acceleration:{self.acceleration:.3f} m/s^2, Energy Remaining: {remaining_energy:.3f} Wh"
        # )

        return [
            self.time_elapsed,
            self.distance_traveled,
            self.speed,
            self.acceleration,
            self.energy_consumed_car,
            self.remaining_energy,
        ]
    



        # for drive_details in drive_results:
        #     print(
        #         f"Time: {drive_details[0]} s, Distance: {drive_details[1]:.2f} m, Speed:{drive_details[2]:.3f} m/s, Acceleration: {drive_details[3]:.1f} m/s^2, Energy Remaining: {drive_details[4]:.3f} Wh"
        #     )



class Solar:
    def __init__(
             
        self, cell_efficiency #= .17
        , area #= 4,
    ):
        self.cell_efficiency = cell_efficiency
        self.area = area

    # Function to calculate the distance between two locations (Haversine formula)
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2

    # Radius of the Earth in kilometers
        earth_radius = 6371

    # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = earth_radius * c

        return distance

    def getincident(self, latitude, longitude, globaltime):
        threshold = 0.3
        cache_key = (latitude, longitude, globaltime)
        # dict  = {}
        res = forecast.radiation_and_weather(
            hours = 15,
            api_key="9iiCFl3djay3KFYH7nWohFtkQ5lslYZE",
            latitude=latitude,
            longitude=longitude,
            output_parameters=['ghi']
            )
        a = res.to_pandas()
        # column_headings = a.columns

# Convert column headings to a list if needed
        # column_headings_list = column_headings.tolist()

# Print the column headings
        # print(column_headings)
        # print(type(a.iloc[0][0]))
        # column_headings = a['ghi'].columns
        # print(column_headings)
        # a['period_end'] = pd.to_datetime(a['ghi'])

    # # Calculate the time difference from 8 AM on the same day and convert it to minutes
        # a['minutes_elapsed'] = (a['period_end'] - a['period_end'].dt.normalize() + pd.to_timedelta('8:00:00')).dt.total_seconds() / 60

        # print(a.head)
        # print(a.head())
        intensity = a.iloc[int((globaltime//3600))]['ghi']
        print(intensity)
        # print(intensity)
        # A = [(26, 63, 1), (25, 63, 2), (24, 63, 3)]
        # A = np.array(A)
        # leftbottom = np.array((latitude, longitude, globaltime))
        # distances = np.linalg.norm(A-leftbottom, axis=1)
        # min_index = np.argmin(distances)
        # intensity = dict[A[min_index]]
        return intensity
    

# Function to retrieve data from the API or cache
    def get_data_from_api_or_cache(self,latitude, longitude, timestamp):
        threshold = 1200
        # Find the closest cache key
        closest_cache_key = None
        min_time_difference = float('inf')

        for cache_key in cache:
            cached_latitude, cached_longitude, cached_timestamp = cache_key
            location_distance = self.calculate_distance(latitude, longitude, cached_latitude, cached_longitude)
            time_difference = abs(timestamp - cached_timestamp)

            if location_distance < 0.01 and time_difference < min_time_difference:
                min_time_difference = time_difference
                closest_cache_key = cache_key

        if closest_cache_key and min_time_difference <= threshold:
            return cache[closest_cache_key]

    # If no close cache key is found or it's not close enough, make an API request
        data = self.getincident(latitude, longitude, timestamp)

    # Update the cache with the new data and timestamp
        cache[(latitude, longitude, timestamp)] = data
        return data

    def calculate_energy(self, time, globaltime, latitude, longitude):
        # Calculate power generated by solar in the path
        power = self.get_data_from_api_or_cache(latitude, longitude, globaltime)*self.cell_efficiency*self.area
        energy = power*time
        return energy




    
# Main program




def main():
    distance = 8000  # meters
    battery_capacity = 7000  # Wh
    wheel_radius = 19 / 39.37  # Convert inches to meters
    mass = 314  # kg
    wheels = 3
    aerodynamic_coef = 0.17
    frontal_area = 1
    zero_speed_crr = 0.003
    avg_m_per_s = 20
    
    route = [[10000, 1*0.015708]
    , [20000, 4*0.015708]
    , [25000, 2*0.015708]
    , [30000, 1*0.015708],[27000, 2*0.015708],[10000, 3*0.015708]
    ]
    

    route_df = pd.read_csv("~/Books/Workings/Agnirath_LVS_Strategy/Strategy/Strategy_V6/data.csv", delimiter=',')
    print(route_df.iloc[0][1])
    distances, slopes = zip(*route)
    distances1 = np.cumsum(distances)
    # distances1 = np.append(0, distances1)
    print(distances1)


    # df = pd.read_csv('route.csv')
    # route = df.values.tolist()        

    # Read the CSV file into a DataFrame
    # df = pd.read_csv('route.csv')  # Replace 'output.csv' with the actual CSV file path
    # route = df.values.tolist()

    # for i in range(len(route)):
    #     route[i].append(route_df.iloc[i, 0] / avg_m_per_s)
    # print("")

    print(route)





    motor = Motor(
        wheel_radius, mass, wheels, aerodynamic_coef, frontal_area, zero_speed_crr
    )
    n_v = 6
    v_max = 30
    car = ElectricCar(motor, distance, battery_capacity, route)
    x_list = []

    
    def obj(x_in):
        solar_panel = Solar(0.17, 4)
        time = 0
        # for i in range(n_v):
        #     time += route[i][0]/x_in[i]
        E_use = 0
        t_arr = []
        for i in range(n_v-1):
            sim1 = car.drive_sim(x_in[i],x_in[i+1], (x_in[i+1] - x_in[i])/(route_df.iloc[i,0]/x_in[i]), battery_capacity-E_use, route[i][0], route[i][1])
            t_arr.append(sim1[0])
            E_use += sim1[4]
            E_use -= solar_panel.calculate_energy(sim1[4], time, route_df.iloc[i,2], route_df.iloc[i,3])/3600
        E_use += car.drive_sim(x_in[n_v-1], 0, 0, battery_capacity-E_use, route_df.iloc[i,0], route_df.iloc[i,1])[4]
        E_use -= solar_panel.calculate_energy(sim1[4], time, route_df.iloc[n_v-1,2], route_df.iloc[n_v-1,3])/3600
        t_arr.append(car.drive_sim(x_in[n_v-1], 0, 0, battery_capacity-E_use, route_df.iloc[i,0], route_df.iloc[i,1])[0])
        # velocity = 0
        # for i in range(n_v+1, n_v+1+n_v):
        #     velocity += x_in[i]*(x_in[i-n_v-1] -  v_max)
        # dist = 0
        # for i in range(n_v+n_v+1, n_v+n_v+1+n_v):
        #     dist += x_in[i]*(route[i-2*n_v-1][0]- x_in[i-2*n_v-1]*t_arr[i-2*n_v-1])
        # x_list.append(x_in)
        time = sum(t_arr)
        # return time
        return time /1000
    
    

    def constraintE(x_in):
        solar_panel = Solar(0.17, 4)
        E_use = 0
        time = 0
        for i in range(n_v-1):
            # print("hi")
            sim1 = car.drive_sim(x_in[i], x_in[i+1], (x_in[i+1] - x_in[i])/(route_df.iloc[i,0]/x_in[i]), battery_capacity-E_use, route_df.iloc[i,0], route_df.iloc[i,1])
            E_use += sim1[4]
            time += sim1[0]
            # print(time)
            # print(solar_panel.calculate_energy(sim1[4], time, route_df.iloc[i,2], route_df.iloc[i,3])/3600)
            # print(solar_panel.calculate_energy(sim1[4], time, route_df.iloc[i,2], route_df.iloc[i,3])/3600)
            E_use -= solar_panel.calculate_energy(sim1[4], time, route_df.iloc[i,2], route_df.iloc[i,3])/3600
        E_use += car.drive_sim(x_in[n_v-1], 0, 0, battery_capacity-E_use, route_df.iloc[i,0], route_df.iloc[i,1])[4]
        E_use -= solar_panel.calculate_energy(sim1[4], time, route_df.iloc[n_v-1,2], route_df.iloc[n_v-1,3])/3600
        # print("Energy_Used:", E_use)
        # print(E_use)
        return -E_use+battery_capacity*0.8  
    


    p0 = np.array([10, 20, 20, 20, 20 ,20])
    p3 = [10, 10, 10, 10, 10, 10, 
          0, 
          0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0]
    #   File "/home/kailash/.local/lib/python3.10/site-packages/pandas/core/indexes/datetimes.py", line 628, in get_loc
    constraints = []

    constraint = {
        'type': 'ineq', 
        'fun' : lambda x: constraintE(x)
    }
    constraints.append(constraint)


    bounds = np.array([(0, 10)] + [(0, 30)] * 4 + [(0, 18)])
    p1 = optimize.minimize(obj, p0, bounds=bounds, method='trust-constr', constraints=constraints, options={'disp':True,'verbose':3, 'maxiter': 1000})
    # , method='trust-constr'
    # , hess = lambda x: np.zeros((19, 19))
    print(p1)





    fig = go.Figure()
    #v = grad_desc([15, 15, 15, 15, 15, 15], 10, route, motor, distance, battery_capacity)
    #print(v)

    x_in = p1.x
    print(x_in)

    E_use_arr = []
    E_use = 0
    time = 0
    solar_panel = Solar(0.17, 4)
    for i in range(n_v-1):
        # print("hi")
        sim1 = car.drive_sim(x_in[i], x_in[i+1], (x_in[i+1] - x_in[i])/(route_df.iloc[i,0]/x_in[i]), battery_capacity-E_use, route_df.iloc[i,0], route_df.iloc[i,1])
        E_use += sim1[4]
        # E_use_arr.append(sim1[4])
        E_use_arr.append(battery_capacity-E_use)
        E_use -= solar_panel.calculate_energy(sim1[4], time, route_df.iloc[i,2], route_df.iloc[i,3])/3600
        time += sim1[0]
            # E_use -= solar_panel.calculate_energy(sim1[4], time, route[2], route[3])
    # E_use_arr.append(car.drive_sim(x_in[n_v-1], 0, 0, battery_capacity-E_use, route[i][0], route[i][1])[4])
    E_use += car.drive_sim(x_in[n_v-1], 0, 0, battery_capacity-E_use, route_df.iloc[i,0], route_df.iloc[i,1])[4]
    E_use -= solar_panel.calculate_energy(sim1[4], time, route_df.iloc[n_v-1,2], route_df.iloc[n_v-1,3])/3600
    E_use_arr.append(battery_capacity-E_use)
    print("E_REMAINING:" , E_use_arr)
    # plt.xlabel('Distance')
    # plt.ylabel('Energy Remaining')
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    x = np.append(0, distances1)
    y = np.append(battery_capacity, E_use_arr)
    cmap = plt.get_cmap('inferno') 
    norm = Normalize(vmin=min(x_in), vmax=max(x_in))
    sm = ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots()

    for i in range(0, 6, 1):
        # r = random.random()
        # b = random.random()
        # g = random.random()
        # color = (r, g, b)
        # print(i, cmap(norm(x_in))[i])
        print(y[i:i+1+1])
        ax.plot(x[i:i+1+1], y[i:i+1+1], c=cmap(norm(x_in))[i], lw=7)
    plt.colorbar(sm, label='Color Legend for velocity')
    plt.xlabel('Distance')
    plt.ylabel('Energy Remaining')
    plt.show()


if __name__ == "__main__":
    main()