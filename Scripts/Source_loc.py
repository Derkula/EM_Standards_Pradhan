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
    e_field = (k*charge/rmag**2)*rhat
    return e_field

def eforce_calc(charge, efield):
    force = charge*efield
    return force


############## QUESTION 1 #################

particle_q = 5e-9
p_r = np.array([0.2,0,0])

pr_mag, pr_hat = mag_hat(p_r)

p_e = e_calc(particle_q, pr_mag, pr_hat)

#print(p_e)  # P1


############# QUESTION 2 ###############

e2_mag = 500

f2 = (-1*e)*e2_mag   # P1,2,
#print(f2)       # P1



############# QUESTION 3 ################

p3_loc = np.array([-0.3, -0.7, -0.2])
loc2 = np.array([0.5, -0.5, -0.5])

pq3 = 6e-9

r3_vec = loc2-p3_loc

r3_mag, r3_hat = mag_hat(r3_vec)

e3_field = e_calc(pq3, r3_mag, r3_hat)


e3_mag, e3_hat = mag_hat(e3_field)

#print(e3_mag) 
#print(e3_field)


############### QUESTION 4 ##################

b4 = np.array([-0.2, -0.4, 0])
q4 = 3e-9

b4_mag, b4_hat = mag_hat(b4)

e4 = e_calc(q4, b4_mag, b4_hat)

#print(e4)

############# QUESTION 5 #################

obs_loc = np.array([.3,0,0])
pc_loc = np.array([.7,0,0])

q5 = 1.5e-9

r5 = pc_loc-obs_loc

r5_mag, r5_hat = mag_hat(r5)

e5 = e_calc(q5, r5_mag, r5_hat)

#print(e5)


########### QUESTION 6 ############

e6_loc = np.array([0.5,0.7,-0.5])
loc6 = np.array([0.5,1.6,-0.5])
q6 = -1*e
r6 = loc6-e6_loc

r6_mag, r6_hat = mag_hat(r6)

e6 = e_calc(q6, r6_mag, r6_hat)
e6_mag, e6_hat = mag_hat(e6)

#print(r6, r6_mag, r6_hat)
#print(e6_mag, e6)


########## QUESTION 7 ###########

e7_mag = 7.8e6 #N/C

e7_half = e7_mag/2
unit_y = np.array([0,1,0])

r = np.sqrt(k*e/e7_half)
print(r)

######## QUESTION 8 ##############

#net elecric field on location x is 0, from a He & proton on opposite sides

r_HeX = np.array([3e-10, 0 , 0])

rhex_mag, rhex_hat = mag_hat(r_HeX)

e_hex = e_calc(2*e, rhex_mag, rhex_hat)

r_ProtX = np.sqrt(k*e/e_hex)

#print(r_ProtX)

# R_elecX is negative r_protx

############ QUESTION 9 ###############


q91 = -3e-9
q92 = 5e-9

loc_9a = np.array([-0.06, -0.04, 0])
loc_q91 = np.array([-0.02, 0.05, 0])
loc_q92 = np.array([0.03, -0.03, 0])

r_9q1a = loc_9a - loc_q91
r_9q2a = loc_9a - loc_q92

r_9q1a_mag, r_9q1a_hat = mag_hat(r_9q1a)

r_9q2a_mag, r_9q2a_hat = mag_hat(r_9q2a)

e_91 = e_calc(q91, r_9q1a_mag, r_9q1a_hat)

e_92 = e_calc(q92, r_9q2a_mag, r_9q2a_hat)

e_9net = e_91 + e_92

#print(e_91, e_92)

#print(e_9net)


######### QUESTION 10 ############


q10_1 = 2e-6
q10_2 = 7e-6
q10_3 = -5e-6

loc10_q1 = np.array([0,0.04,0])
loc10_q2 = np.array([0,0,0])
loc10_q3 = np.array([0.03,0,0])

r10_21 = loc10_q1
r10_31 = loc10_q1 - loc10_q3

r10_21_mag, r10_21_hat = mag_hat(r10_21)
r10_31_mag, r10_31_hat = mag_hat(r10_31)

e10_21 = e_calc(q10_2, r10_21_mag, r10_21_hat)
e10_31 = e_calc(q10_3, r10_31_mag, r10_31_hat)

e10_2131 = e10_21 + e10_31
#print(e10_2131)

f10_2131 = q10_1 * e10_2131
#print(f10_2131)

loc10_a = np.array([0.03,0.04,0])

r10_q1a = loc10_q3
r10_q2a = loc10_a
r10_q3a = loc10_q1


r_q1a_mag, r_q1a_hat = mag_hat(r10_q1a)
r_q2a_mag, r_q2a_hat = mag_hat(r10_q2a)
r_q3a_mag, r_q3a_hat = mag_hat(r10_q3a)

e10_1a = e_calc(q10_1, r_q1a_mag, r_q1a_hat)
e10_2a = e_calc(q10_2, r_q2a_mag, r_q2a_hat)
e10_3a = e_calc(q10_3, r_q3a_mag, r_q3a_hat)

e10_anet = e10_1a + e10_2a + e10_3a

#print(e10_anet)
m10 = 6.646e-27

f10_qe = 2*e*e10_anet
f10_acc = f10_qe/m10

#print(f10_acc)


############ QUESTION 11 ##############


q11_pos = 7e-9
q11_neg = -1 * q11_pos

s = 0.003

loc11_qpos = np.array([0,s/2,0])
loc11_qneg = -1*loc11_qpos

loc11a = np.array([0,0.05,0])

r11_pa = loc11a-loc11_qpos
r11_na = loc11a-loc11_qneg

r11pa_mag, r11pa_hat = mag_hat(r11_pa)
r11na_mag, r11na_hat = mag_hat(r11_na)

e11_pa = e_calc(q11_pos, r11pa_mag, r11pa_hat)
e11_na = e_calc(q11_neg, r11na_mag, r11na_hat)

e11net_a = e11_pa + e11_na 

#print(e11net_a)

loc11b = np.array([0.05,0,0])

r11_pb = loc11b-loc11_qpos
r11_nb = loc11b-loc11_qneg

r11pb_mag, r11pb_hat = mag_hat(r11_pb)
r11nb_mag, r11nb_hat = mag_hat(r11_nb)

e11_pb = e_calc(q11_pos, r11pb_mag, r11pb_hat)
e11_nb = e_calc(q11_neg, r11nb_mag, r11nb_hat)

e11net_b = e11_pb + e11_nb
#print(e11net_b)


################# QUESTION 12 ##################

q12_1 = 13e-9
q12_2 = 8e-9

r_q1apos = np.array([0.24,-0.002,0])
r_q1apos_mag, r_q1apos_hat = mag_hat(r_q1apos)
e_q1apos = e_calc(q12_1, r_q1apos_mag, r_q1apos_hat)

r_q1aneg = np.array([0.24,0.002,0])
r_q1aneg_mag, r_q1aneg_hat = mag_hat(r_q1aneg)
e_q1aneg = e_calc(-1*q12_1, r_q1aneg_mag, r_q1aneg_hat)

r_q2apos = np.array([0,15.9985,0])
r_q2apos_mag, r_q2apos_hat = mag_hat(r_q2apos)
e_q2apos = e_calc(q12_2, r_q2apos_mag, r_q2apos_hat)

r_q2aneg = np.array([0,16.0015,0])
r_q2aneg_mag, r_q2aneg_hat = mag_hat(r_q2aneg)
e_q2aneg = e_calc(-1*q12_2, r_q2aneg_mag, r_q2aneg_hat)

edp_1 = e_q1apos + e_q1aneg
edp_2 = e_q2apos + e_q2aneg 
e12_net = e_q1apos + e_q1aneg + e_q2apos + e_q2aneg

print(edp_1)
print(edp_2)
print(e12_net)





s12_1 = 0.004
s12_1_center = s12_1/2


s12_2 = 0.003
s12_2_center = s12_2/2
