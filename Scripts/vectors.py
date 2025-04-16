import numpy as np

a = np.array([-1e10,7e10,-6e10])
mag_a = np.linalg.norm(a)
print(mag_a)

b = np.array([3e10,-8e10,6e10])
mag_b = np.linalg.norm(b)
print(mag_b)

unit_a = a/mag_a
unit_b = b/mag_b
print("Unit a: ", unit_a)
print("Unit b; ", unit_b)

c = a-b
print(c)
mag_c = np.linalg.norm(c)
print(mag_c)

unit_c = c/mag_c

print(unit_c)
#Position of a particle, time of travel * veector
#pos = 0.018*a
#print(pos)