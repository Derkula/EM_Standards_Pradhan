import numpy as np
import scipy.constants as const

# Constants
e0= const.epsilon_0
e = const.elementary_charge
k = 1/(4*np.pi*e0)


def mag_hat(vector):
    # Calculate magnitude and unit vectors
    magnitude = np.linalg.norm(vector)
    hat = vector/magnitude
    return magnitude, hat

def e_calc(charge, rmag, rhat):
    # Calculate electric field, EQN: (kq/|r|^2)*rhat
    e = (k*charge/rmag**2)*rhat
    return e

def eforce_calc(charge, efield):
    force = charge*efield
    return force

############# QUESTION 1 ###############3

q1 = -2.5e-6
q2 = 3.5e-6
q3 = -5e-6
q4 = -2e-9  # P9

#### Calculating net electric field at q3
r13 = np.array([0.04,-0.03,0])
r23 = np.array([0,-0.03,0])     # P2
r1a = np.array([0,-0.03,0])     # P5
r2a = np.array([-0.04,-0.03,0]) # P6
r3a = np.array([-0.04,0,0])     # P7


r13_mag, r13_hat = mag_hat(r13) # P3
r23_mag, r23_hat = mag_hat(r23) # P2
r1a_mag, r1a_hat = mag_hat(r1a) # P5
r2a_mag, r2a_hat = mag_hat(r2a) # P6
r3a_mag, r3a_hat = mag_hat(r3a) # P7

print("sure ",r13, r13_mag, r13_hat)
e13 = e_calc(q1, r13_mag, r13_hat) # P3
e23 = e_calc(q2, r23_mag, r23_hat) # P3
e1a = e_calc(q1, r1a_mag, r1a_hat) # P5
e2a = e_calc(q2, r2a_mag, r2a_hat) # P6
e3a = e_calc(q3, r3a_mag, r3a_hat) # P7

e3_net = e13 + e23 # P3
ea_net = e1a + e2a + e3a # P7



f13 = eforce_calc(q3, e13) # P4
f23 = eforce_calc(q3, e23) # P4

f_net = f13 + f23 # P4
f2_net = q3*e3_net # P4 (alt)

fa_net = q4*ea_net # P9

print("electric field: ", e13) 
#print("Net electric field: ", e_net)  # P3
#print("Net force: ", f_net)            #p4
#print("f2: ", f2_net)                  #p4
#print("Electric field at A: ", e_net)  # P5,6,7,8
#print("Force on A with charge: ", fa_net) # P9 


############# QUESTION 2 ###############

Fe_q = 3*e   
Cl_q = -1*e  
q_a = Cl_q

r = np.array([148e-9,0,0])
r_FeCl_vec = np.array([400e-9,0,0])  
r_FeA = r_FeCl_vec - r     # P1  Was using 2.52e-9 instead of 252e-9
r_ClA = -1*r               # P1
r_FeB = r_FeCl_vec + r     # P2
r_ClB = r                  # P2 

r_fc_mag, r_fc_hat = mag_hat(r_FeCl_vec)    # P1
r_fa_mag, r_fa_hat = mag_hat(r_FeA)         # P1
r_ca_mag, r_ca_hat = mag_hat(r_ClA)         # P1
r_fb_mag, r_fb_hat = mag_hat(r_FeB)         # P2             
r_cb_mag, r_cb_hat = mag_hat(r_ClB)         # P2


e_fa = e_calc(Fe_q, r_fa_mag, r_fa_hat) # P1
e_ca = e_calc(Cl_q, r_ca_mag, r_ca_hat) # P1
e_fb = e_calc(Fe_q, r_fb_mag, r_fb_hat) # P2
e_cb = e_calc(Cl_q, r_cb_mag, r_cb_hat) # P2


e_a_net = e_fa + e_ca   # P1
e_b_net = e_fb + e_cb   # P2

f_a = q_a*e_a_net       # P3

#print("Net electric field at a: ", e_b_net)   # P1, 2
#print("Force at A: ", f_a)                    # P3

############ QUESTION 3 ##############

rod_q = -9e-8
length = 1.5
piece = length/8
piece_q = rod_q/8

origin = np.array([0,0,0])
a_loc = np.array([0.7,0,0])
top = np.array([0,length/2,0])
p_center = np.array([0,piece/2,0])
bottom = -1*top

seg_1 = top - p_center  # P1

r_1a = a_loc - seg_1                        # P3
r1a_mag, r1a_hat = mag_hat(r_1a)            # P3
e_1a = e_calc(piece_q, r1a_mag, r1a_hat)    # P3


#print("Center of Seg 1: ", seg_1)    # P1
#print(piece_q)                       # P2
#print("Distance 1 to A: ", r_1a)     # P3
#print("E field from seg 1: ", e_1a)   # P3


################ QUESTION 5 ##################

t_length = 0.14
t_width = 0.018
tape_q = 3e-9

t_seg = t_length/3
t_seg_center = t_seg/2
t_seg_q = tape_q/3

a5_loc = np.array([0,0.02,0])

t_seg_end = np.array([t_length/2,0,0])
t_seg_center = np.array([t_seg_center,0,0])
t_seg3 = t_seg_end - t_seg_center
t_seg1 = -1 * t_seg3

t_r1a = a5_loc - t_seg1
t_r3a = a5_loc - t_seg3

a5_mag, a5_hat = mag_hat(a5_loc)
t_r1a_mag, t_r1a_hat = mag_hat(t_r1a)
t_r3a_mag, t_r3a_hat = mag_hat(t_r3a)

t_e1 = e_calc(t_seg_q, t_r1a_mag, t_r1a_hat)
t_e2 = e_calc(t_seg_q, a5_mag, a5_hat)
t_e3 = e_calc(t_seg_q, t_r3a_mag, t_r3a_hat)

t_enet = t_e1 + t_e2 + t_e3

#print(t_e1, t_e2, t_e3)        # P1,2,3
#print(t_enet)                  # P4


