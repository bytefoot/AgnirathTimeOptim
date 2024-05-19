
# Model Settings
ModelMethod = "trust-constr"

# ---------------------------------------------------------------------------------------------------------
# Car Data

# Battery
BatteryCapacity = 3000  # Wh
DeepDischargeCap = 0.20  # 20%

# Physical Attributes
WheelRadius = 19 / 39.37  # Convert `in` to `m``
Mass = 314  # kg
Wheels = 3

# Resistive Coeff
CDA = 0.092
ZeroSpeedCrr = 0.003

# Solar Panel Data
PanelArea = 4  # m^2
PanelEfficiency = 0.17

# ---------------------------------------------------------------------------------------------------------
# Physical Constants
AirDensity = 1.225  # kg/m^3
g = GravityAcc = 9.8  # m/s^2

# ---------------------------------------------------------------------------------------------------------
# Car Constraints
MaxVelocity = 36
