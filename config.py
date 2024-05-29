
# Model Settings
ModelMethod = "trust-constr"
InitialGuessVelocity = 35

RaceStartTime = 9 * 3600  # 9:00 am
RaceEndTime = (12 + 6) * 3600  # 6:00 pm

# ---------------------------------------------------------------------------------------------------------
# Car Data

# Battery
BatteryCapacity = 3055  # Wh
DeepDischargeCap = 0.20  # 20%

# Physical Attributes
WheelRadius = 0.2785
Mass = 270 # kg
Wheels = 3

# Resistive Coeff
CDA = 0.092
ZeroSpeedCrr = 0.0045

# Solar Panel Data
PanelArea = 6  # m^2
PanelEfficiency = 0.19

# ---------------------------------------------------------------------------------------------------------
# Physical Constants
AirDensity = 1.192 # kg/m^3
g = GravityAcc = 9.81 # m/s^2

# ---------------------------------------------------------------------------------------------------------
# Car Constraints
MaxVelocity = 36  # m/s
MaxAcceleration  = 1.1  # m/s^2