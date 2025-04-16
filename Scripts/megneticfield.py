import numpy as np 
import scipy.constants as const

b_field = np.array([0,6.626e-23,0])
q = -1 * const.elementary_charge
v = np.array([300,0,0])

f_field = np.cross(q*v,b_field)
print(f_field)