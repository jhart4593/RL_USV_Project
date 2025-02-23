import math
import numpy as np
from config import config as cfg
from eval_config import eval_config as e_cfg
from fossen_gnc import ssa
from utils import PID_fixed, LOS_guidance, get_current
from otter import otter


def step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c):
    """
    step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c) takes in vehicle, data,
    sampletime, yaw error, Vc, and beta_c, where data = [eta,nu,u_control,u_actual,tau_X,tau_N]
    and outputs an updated data list.
    """

    tau_N = data[17]
    tau_X = data[16]
    eta = data[0:6]
    nu = data[6:12]
    u_control = data[12:14]
    u_actual = data[14:16]

    tau_N = PID_fixed(tau_N,yaw_err)

    u_control = vehicle.controlAllocation(tau_X,tau_N)

    [eta,nu,u_actual,nu_dot] = (
        vehicle.dynamics(eta,nu,u_actual,u_control,sampleTime,Vc,beta_c)
    )

    new_data = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + [tau_X,tau_N])

    return new_data


def step_otter_fossen(vehicle,data,sampleTime,yaw_err,Vc,beta_c):
    """
    step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c) takes in vehicle, data,
    sampletime, yaw error, Vc, and beta_c, where data = [eta,nu,u_control,u_actual,tau_X,tau_N]
    and outputs an updated data list using fossen's heading Autopilot method of otter to 
    set the control input.
    """

    tau_N = data[17]
    tau_X = data[16]
    eta = data[0:6]
    nu = data[6:12]
    u_control = data[12:14]
    u_actual = data[14:16]

    u_control, tau_N = vehicle.headingAutopilot(eta,nu,cfg["sim_dt"])

    [eta,nu,u_actual,nu_dot] = (
        vehicle.dynamics(eta,nu,u_actual,u_control,sampleTime,Vc,beta_c)
    )

    new_data = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + [tau_X,tau_N])

    return new_data


def sim_otter_course_angle():

    num_steps = e_cfg["eval_steps"]
    sampleTime = cfg["sim_dt"]
    disturbance_angle = e_cfg["water_curr_angle"]
    target_course = e_cfg["target_course_angle"]

    # set constant surge control input
    tau_X = cfg["tau_X"]

    # set water current velocity and angle from eval_config file
    Vc = e_cfg["water_curr_vel"]

    def reset():
        # set time and target course
        curr_vel = e_cfg["water_curr_vel"]

        # Reset vehicle instance
        vehicle = otter(tau_X=tau_X)

        # set initital state of USV
        init_course_angle = e_cfg["init_crs_angle"]
        eta = np.array([0,0,0,0,0,init_course_angle])
        nu = np.array([0, 0, 0, 0, 0, 0], float) 
        nu_dot = np.array([0, 0, 0, 0, 0, 0], float)
        u_actual = np.array([0, 0], float)    
        u_control = np.array([0, 0], float)  

        # reset other values
        yaw_err = [0,0,(target_course - eta[5])]
        tau_N = 0

        return curr_vel,vehicle,eta,nu,u_actual,u_control,yaw_err,tau_N
    

    simData_total = np.empty([num_steps+1,0],float)

    for kk in range(len(disturbance_angle)):
        curr_vel,vehicle,eta,nu,u_actual,u_control,yaw_err,tau_N = reset()
        data = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + [tau_X,tau_N])
        beta_c = disturbance_angle[kk]
        [current_mag,current_angle] = get_current(curr_vel,cfg["mu"],cfg["water_curr_vel_high"],cfg["Vmin"],beta_c)
        simData = np.array(eta[5])

        for ii in range(num_steps):
            t = ii * sampleTime
            if t < 1:
                Vc = 0
            else:
                Vc = current_mag[ii][0]

            data = step_otter(vehicle,data,sampleTime,yaw_err,Vc,current_angle[ii])
            eta = data[0:6]
            yaw_err.append(target_course - eta[5])

            simData= np.vstack((simData,np.array(eta[5])))
        
        simData_total = np.hstack((simData_total,simData))

    simTime = []
    for jj in range(simData.shape[0]):
        t = jj * sampleTime
        simTime.append(t)

    return simData_total, simTime


def sim_otter_LOS(path_x,path_y):

    path = np.vstack((np.array(path_x),np.array(path_y)))
    acc_rad = e_cfg["acc_rad"]
    over_dist = e_cfg["over_dist"]
    num_steps = e_cfg["eval_steps_LOS"]
    sampleTime = cfg["sim_dt"]
    tau_X = cfg["tau_X"]
    tau_N = 0

    # set water current velocity and angle from eval_config file
    curr_vel = e_cfg["LOS_Vc"]
    beta_c = e_cfg["LOS_beta_c"]

    vehicle = otter(tau_X=tau_X)


    # set initital state of USV
    init_course_angle = e_cfg["LOS_init_crs_angle"]
    xpos = path[0,0]; ypos = path[1,0]
    eta = np.array([xpos,ypos,0,0,0,init_course_angle])
    nu = np.array([0, 0, 0, 0, 0, 0], float) 
    nu_dot = np.array([0, 0, 0, 0, 0, 0], float)
    u_actual = np.array([0, 0], float)    
    u_control = np.array([0, 0], float)  
    yaw_error = e_cfg["LOS_init_yaw_err"]
    path_error = e_cfg["LOS_init_path_err"]
    path_idx = 0
    goal = 0
    t = 0

    yaw_err = [0,0,yaw_error]

    data = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + [tau_X,tau_N])

    simData = np.array(
    list(eta) + list(nu) + list(u_control) + list(u_actual)
    + [t] + [yaw_error,path_error] + [tau_X,tau_N]
    )

    for ii in range(num_steps):
        t = ii * sampleTime
        if t < 1:
                Vc = 0
        else:
            Vc = curr_vel

        data = step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c)

        tau_N = data[17]
        tau_X = data[16]
        eta = data[0:6]
        nu = data[6:12]
        u_control = data[12:14]
        u_actual = data[14:16]

        [yaw_error,path_error,path_idx,goal] = LOS_guidance(path_idx,path,eta,acc_rad,over_dist)

        yaw_err.append(yaw_error)

        # Store simulation data
        simData = np.vstack(
            (simData,np.array(list(eta) + list(nu) + list(u_control) 
                + list(u_actual) + [t] + [yaw_error,path_error]
                + [tau_X,tau_N]))
        )

        if goal != 0:
            break

    simTime = []
    for jj in range(simData.shape[0]):
        t = jj * sampleTime
        simTime.append(t)

    return simTime, simData