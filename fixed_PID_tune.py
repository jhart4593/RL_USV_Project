import math
import numpy as np
from config import config as cfg
from eval_config import eval_config as e_cfg
from fossen_gnc import ssa
from matplotlib import pyplot as plt
from otter import otter


def PID_tune(coeffs, Tn, yaw_err):
    """
    Tn = PID_tune(coeffs, Tn, yaw_err) takes in the PID coefficient values, current tau_N, and yaw_err list
    over time and outputs the new tau_N.
    """

    #PID controller
    kp = coeffs[0]
    ki = coeffs[1]
    kd = coeffs[2]

    P = yaw_err[-1] - yaw_err[-2]
    I = yaw_err[-1]
    D = yaw_err[-1] - 2*yaw_err[-2] + yaw_err[-3]

    delTn = kp*P + ki*I + kd*D

    Tn = delTn + Tn

    return Tn


def step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c,coeffs):
    """
    step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c,coeffs) takes in vehicle, data,
    sampletime, yaw error, Vc, and beta_c, and PID coefficients, where 
    data = [eta,nu,u_control,u_actual,tau_X,tau_N] and outputs an updated data list.
    """

    tau_N = data[17]
    tau_X = data[16]
    eta = data[0:6]
    nu = data[6:12]
    u_control = data[12:14]
    u_actual = data[14:16]

    tau_N = PID_tune(coeffs,tau_N,yaw_err)

    u_control = vehicle.controlAllocation(tau_X,tau_N)

    [eta,nu,u_actual,nu_dot] = (
        vehicle.dynamics(eta,nu,u_actual,u_control,sampleTime,Vc,beta_c)
    )

    new_data = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + [tau_X,tau_N])

    return new_data


def tune_PID():
    """
    simData,simTime = tune_PID() simulates the otter vehicle over time for a range
    of fixed PID coefficients and returns the course angle data for each combination as
    well as the time.
    """

    num_steps = e_cfg["eval_steps"]
    sampleTime = cfg["sim_dt"]
    disturbance_angle = e_cfg["water_curr_angle"]
    target_course = e_cfg["target_course_angle"]

    # set constant surge control input
    tau_X = cfg["tau_X"]

    # set water current velocity and angle from eval_config file
    Vc = e_cfg["water_curr_vel"]
    beta_c = math.pi/2

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
        yaw_err = [0,0,ssa((target_course - eta[5]))]
        tau_N = 0

        return curr_vel,vehicle,eta,nu,u_actual,u_control,yaw_err,tau_N
    

    simData_total = np.empty([num_steps+1,0],float)

    for ii in np.linspace(0,4,5):

        for jj in np.linspace(0,1,5):

            for kk in np.linspace(0,80,5):

                curr_vel,vehicle,eta,nu,u_actual,u_control,yaw_err,tau_N = reset()
                data = np.array(list(eta) + list(nu) + list(u_control) + list(u_actual) + [tau_X,tau_N])
                simData = np.array(eta[5])

                for mm in range(num_steps):
                    t = mm * sampleTime
                    if t < 1:
                        Vc = 0
                    else:
                        Vc = curr_vel

                    data = step_otter(vehicle,data,sampleTime,yaw_err,Vc,beta_c,[kk,jj,ii])
                    eta = data[0:6]
                    yaw_err.append(ssa(target_course - eta[5]))

                    simData= np.vstack((simData,np.array(eta[5])))
        
                simData_total = np.hstack((simData_total,simData))

    simTime = []
    for nn in range(simData.shape[0]):
        t = nn * sampleTime
        simTime.append(t)

    return simData_total, simTime


if __name__ == "__main__":

    simData,simTime = tune_PID()

    def R2D(value):  # radians to degrees
        return value * 180 / math.pi

    # Course angle plots over time
    psi = R2D(simData)
    half = psi.shape[0]//2
    psi1 = psi[0:half,:]
    psi2 = psi[half:,:]
    time1 = simTime[0:half]
    time2 = simTime[half:]
    psi_d1 = np.full(len(time1),e_cfg["target_course_angle"])
    psi_d2 = np.full(len(time2),e_cfg["target_course_angle"])

    plt.figure(1)
    for ii in range(psi1.shape[1]):
        plt.plot(time1,psi1[:,ii])

    plt.plot(time1,psi_d1,'--k')
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle (deg)')
    plt.xlim(0,90)
    plt.ylim(-45, 180)
    # plt.title('Course Angle over Episode')
    # plt.legend(['Course angle', 'Target course angle'])
    plt.grid(True)

    plt.figure(2)
    for jj in range(psi2.shape[1]):
        d = psi2[:,jj]
        if all(not abs(num)>5 for num in d):
            plt.plot(time2,d,label = str(jj))

    plt.plot(time2,psi_d2,'--k')
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle (deg)')
    plt.xlim(90,180)
    plt.ylim(-5, 5)
    # plt.title('Course Angle over Episode')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(simData.shape)
    print(psi.shape)