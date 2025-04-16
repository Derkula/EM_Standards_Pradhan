import numpy as np
import scipy.constants as const

# Constants
e0= const.epsilon_0
e = const.elementary_charge
k = 1/(4*np.pi*e0)


def vectorize(final, initial):
    # Input variables are not vectorized. This function puts them in vector format, then calculates
    # The distance between the points, with the first entry as the final location. 
    # Then will calculate the magnitude and unit vector for that distance
    # Returns the distance vector, magnitude, and unit vector

    f = np.array([final])
    i = np.array([initial])
    distance = f - i
    magnitude = np.linalg.norm(distance)
    hat = distance/magnitude
    return distance, magnitude, hat

def e_calc(charge, rmag, rhat):
    # Calculate electric field, EQN: (kq/|r|^2)*rhat

    e = (k*charge/rmag**2)*rhat
    return e

electron = -1*e

# Inputting the coordinates of each point of interest.
origin = [0,0,0]                  # Origin location
e_loc = [0.004, -0.026, -0.402]   # Electron location
a_loc = [0.062, 0.033, -0.402]    # Observed location


# Vector, magnitude, and unit for distance from electron to origin
r_e_origin, r_eo_mag, r_eo_hat = vectorize(origin, e_loc) 

# Vector, magnitude, and unit for distance from electron to observed point
r_e_a, r_ea_mag, r_ea_hat = vectorize(a_loc, e_loc)

# Calculating the electric field at the origin due to the electron
e1 = e_calc(electron, r_eo_mag, r_eo_hat)

# Calculating the electric field at the observed location due to the electron
e2 = e_calc(electron, r_ea_mag, r_ea_hat)

# Outputs
print(r_e_origin, r_eo_mag, r_eo_hat)
print(r_e_a, r_ea_mag, r_ea_hat)
print("Electric Field at Origin: ", e1)
print("Electric Field at Observed Location: ", e2)

print(k*electron)