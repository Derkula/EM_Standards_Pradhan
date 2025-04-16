import numpy as np
import scipy.constants as const

e_field = np.array([1000,0,0])
e_mag = np.linalg.norm(e_field)
e_hat = e_field/e_mag

#print("E mag: ", e_field)

#print('E hat: ', e_field)
q = 2*1.6e-19
f_qe= q*e_field
f_mag = np.linalg.norm(f_qe)
f_hat = f_qe/f_mag

print("force E_field :", f_qe)

print(f_hat)

# Proton accelleration
proton_mass = const.proton_mass
e_charge = const.elementary_charge
a = 9e11
E = (proton_mass*a)/e_charge

#print(E)

# magnitude of e field on electron due to proton in hydrogen
r = np.array([0.5e-10,0,0])
r_mag = np.linalg.norm(r)
r_hat = r/r_mag
k = 1/(4*np.pi*const.epsilon_0)
e_prot = (k*e_charge/r**2)*r_hat
print(e_prot)