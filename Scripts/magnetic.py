import numpy as np
import scipy.constants as const

e = const.elementary_charge

def mag_calc(charge,velocity,mag_f):
    mag_force = np.cross(charge*velocity,mag_f)
    return mag_force

v1 = np.array([8e5,0,0])
b1 = np.array([0,0.25,0])

mf1 = mag_calc(e, v1, b1)

#print(mf1)

v2 = np.array([6e5,0,0])
b2 = np.array([0,-0.32,0])

mf2 = mag_calc(-1*e, v2, b2)

print(mf2)