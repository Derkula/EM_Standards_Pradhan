import scipy.constants as const
q = 0.4e-7
e0 = const.epsilon_0
r = 0.11
l = 0.6
pi = 2*const.pi
holder = e0*r*l
denom = pi*holder
print (pi)
field = q/denom

print(e0)

e_charge = const.elementary_charge
mass_e = const.electron_mass
mass_p = const.proton_mass

f_neg = -6*e_charge*field
f_pos = 6*e_charge*field
print ("Mag E = ", field)
f_net = f_neg + f_pos
alpha = 1.96e-40
s_plus= alpha*field/(6*e_charge)
s_neg = alpha*field/(-6*e_charge)
s_tot = s_plus - s_neg
print(s_tot)

print(f_net)