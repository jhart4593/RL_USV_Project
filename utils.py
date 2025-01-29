import math
import numpy as np
from config import config as cfg
from eval_config import eval_config as e_cfg
from fossen_gnc import ssa


def PID(action, Tn, yaw_err):
    """
    [Tn,delTn,cont_coeffs] = PID(action,Tn,yaw_err) takes in the action values, current tau_N, and yaw_err list
    over time and outputs the new tau_N, control incremental tau_N command, and the terms that were multiplied
    by the P,I, and D coefficients respectively.
    """

    # rudder PID controller
    kp = action[0]
    ki = action[1]
    kd = action[2]

    P = yaw_err[-1] - yaw_err[-2]
    I = yaw_err[-1]
    D = yaw_err[-1] - 2*yaw_err[-2] + yaw_err[-3]

    cont_coeffs = [P, I, D]

    delTn = kp*P + ki*I + kd*D

    Tn = delTn + Tn

    return Tn, delTn, cont_coeffs


def get_obs(yaw_err, nu, nu_dot, u_actual, delTn, cont_coeffs):
    """
    s_t = get_obs returns the observation array given the current state of the USV
    """
    yaw_e = yaw_err[-1]

    vel = [nu[0],nu[1],nu[5]]
    acc = [nu_dot[0],nu_dot[1],nu_dot[5]]

    s_t = np.array([yaw_e] + vel + acc + list(u_actual) + [delTn] + cont_coeffs)

    return np.float32(s_t)


def get_obs_norm(yaw_err, nu, nu_dot, u_actual, delTn, cont_coeffs):
    """
    s_t = get_obs_norm returns the normalized observation array given the current state of the USV
    """
    yaw_err_lim = cfg["yaw_err_lim"]
    vel_lim = cfg["vel_lim"]
    acc_lim = cfg["acc_lim"]
    ang_vel_lim = cfg["ang_vel_lim"]
    ang_acc_lim = cfg["ang_acc_lim"]
    prop_rpm_max = cfg["prop_rpm_max"]
    prop_rpm_min = cfg["prop_rpm_min"]
    delTn_lim = cfg["delTn_lim"]
    P_lim = cfg["P_limit"]
    I_lim = cfg["I_limit"]
    D_lim = cfg["D_limit"]

    yaw_e = yaw_err[-1]/yaw_err_lim

    vel = [nu[0]/vel_lim,nu[1]/vel_lim,nu[5]/ang_vel_lim]
    acc = [nu_dot[0]/acc_lim,nu_dot[1]/acc_lim,nu_dot[5]/ang_acc_lim]

    u_norm = (u_actual - prop_rpm_min)/(prop_rpm_max - prop_rpm_min)*2 - 1

    P_norm = cont_coeffs[0]/P_lim
    I_norm = cont_coeffs[1]/I_lim
    D_norm = cont_coeffs[2]/D_lim

    s_t = np.array([yaw_e] + vel + acc + list(u_norm) + [delTn/delTn_lim] + [P_norm,I_norm,D_norm])

    return np.float32(s_t)

def LOS_guidance(Pk_idx,path,eta,acc_rad,over_dist):
    '''
    [yaw_e,path_e,Pk_idx,goal]=LOS_guidance(Pk_idx,path,eta,acc_rad,over_dist) takes in,start waypt index, 
    desired path coordinates, eta vector, acceptance radius, and overshoot distance, and outputs the yaw error,
    path error, starting waypt index, and whether the end waypt was reached.
    '''
    # calculate distance between USV and next waypt
    Pk = path[:,Pk_idx]
    Pk1 = path[:,Pk_idx+1]
    x_end_dist = Pk1[0] - eta[0]
    y_end_dist = Pk1[1] - eta[1]
    dist_to_next = math.sqrt((x_end_dist)**2 + (y_end_dist)**2)

    goal = 0
    # if USV is within acceptance radius or past overshoot distance, iterate waypts
    # if waypt is reached and it's the last waypt, return goal reached unless it 
    # was reached due to being over the overshoot distance
    if (dist_to_next <= acc_rad):
        Pk_idx += 1
        if (Pk_idx+1) == path.shape[1]:
            goal = 1
        else:
            Pk = Pk1
            Pk1 = path[:,Pk_idx+1]

    if (dist_to_next > over_dist):
        Pk_idx += 1
        if (Pk_idx+1) == path.shape[1]:
            goal = 2
        else:
            Pk = Pk1
            Pk1 = path[:,Pk_idx+1]

    # calculate yaw and path errors using LOS guidance law
    delx = Pk1[0] - Pk[0]; dely = Pk1[1] - Pk[1]
    yaw_p = math.atan2(dely,delx)

    path_e = -(eta[0] - Pk[0])*math.sin(yaw_p) + (eta[1] - Pk[1])*math.cos(yaw_p)

    yaw_d = yaw_p - math.atan2(path_e,e_cfg["LOS_del"])
    yaw_e = ssa(yaw_d - eta[5])

    return yaw_e, path_e, Pk_idx, goal


def PID_fixed(Tn, yaw_err):
    """
    [Tn,delTn,cont_coeffs] = PID(action,Tn,yaw_err) takes in the action values, current tau_N, and yaw_err list
    over time and outputs the new tau_N, control incremental tau_N command, and the terms that were multiplied
    by the P,I, and D coefficients respectively.
    """

    # rudder PID controller
    kp = e_cfg["PID_fixed_coeff"][0]
    ki = e_cfg["PID_fixed_coeff"][1]
    kd = e_cfg["PID_fixed_coeff"][2]

    P = yaw_err[-1] - yaw_err[-2]
    I = yaw_err[-1]
    D = yaw_err[-1] - 2*yaw_err[-2] + yaw_err[-3]

    delTn = kp*P + ki*I + kd*D

    Tn = delTn + Tn

    return Tn



# test the functions
if __name__ == "__main__":
  
    action = [25,2,1]
    yaw_err = [0,0,math.radians(180)]
    Tn = 100
    nu = np.array([2,1,3,0.2,0.5,0.7])
    nu_dot = np.array([1,1.5,0,0,0,2])
    u_actual = np.array([45,60])
    delTn = 15
    cont_coeffs = [2,3,6]

    [Tn1,delTn1,cc] = PID(action,Tn,yaw_err)
    print(Tn1,delTn1,cc)

    st = get_obs(yaw_err,nu,nu_dot,u_actual,delTn,cont_coeffs)
    print(st)

    st_norm = get_obs_norm(yaw_err,nu,nu_dot,u_actual,delTn,cont_coeffs)
    print(st_norm)

    Pk_idx = 1
    path = np.vstack((np.array([400,500,475.53]),np.array([-10,0,154.51])))
    eta = [500,0,0,0,0,math.pi/2]
    acc_rad = 0.8
    over_dist = 200
    [yaw_e,path_e,Pk_idx,goal]=LOS_guidance(Pk_idx,path,eta,acc_rad,over_dist)
    print([yaw_e,path_e,Pk_idx,goal])
