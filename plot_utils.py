import math
import matplotlib.pyplot as plt
import numpy as np
from fossen_gnc import ssa

def R2D(value):  # radians to degrees
    return value * 180 / math.pi


def plotCourseAngle(simData,simTime,simData_f,simTime_f,target_course):
    """
    plotCourseAngle(simData,simTime,simData_fixed,simTime_fixed,target_course) takes in the simData 
    and simTime for the PPO policy as well as the fixed PID controller, and target course angle, 
    and plots the course angle over time.
    """
    # Course angle plots over time - PPO policy
    psi = R2D(np.average(ssa(simData),axis=1,keepdims=True))
    psi_std = R2D(np.std(ssa(simData),axis=1,keepdims=True))
    half = len(psi)//2
    psi1 = psi[0:half].T[0]
    psi2 = psi[half:].T[0]
    psi_std1 = psi_std[0:half].T[0]
    psi_std2 = psi_std[half:].T[0]
    time1 = simTime[0:half]
    time2 = simTime[half:]
    psi_d1 = np.full(len(psi1),target_course)
    psi_d2 = np.full(len(psi2),target_course)

    # Course angle plots over time - fixed PID policy
    psi_f = R2D(np.average(ssa(simData_f),axis=1,keepdims=True))
    psi_std_f = R2D(np.std(ssa(simData_f),axis=1,keepdims=True))
    half_f = len(psi_f)//2
    psi1_f = psi_f[0:half_f].T[0]
    psi2_f = psi_f[half_f:].T[0]
    psi_std1_f = psi_std_f[0:half_f].T[0]
    psi_std2_f = psi_std_f[half_f:].T[0]
    time1_f = simTime_f[0:half_f]
    time2_f = simTime_f[half_f:]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(time1,psi1)
    plt.plot(time1_f,psi1_f)
    plt.plot(time1,psi_d1,'--k')
    plt.fill_between(time1,psi1-psi_std1,psi1+psi_std1,alpha=0.2)
    plt.fill_between(time1_f,psi1_f-psi_std1_f,psi1_f+psi_std1_f,alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle (deg)')
    plt.xlim(0,80)
    plt.ylim(-45, 180)
    # plt.title('Course Angle over Episode')
    plt.legend(['PPO','PID','Target course angle'])
    plt.grid(True)

    plt.subplot(212)
    plt.plot(time2,psi2)
    plt.plot(time2_f,psi2_f)
    plt.plot(time2,psi_d2,'--k')
    plt.fill_between(time2,psi2-psi_std2,psi2+psi_std2,alpha=0.2)
    plt.fill_between(time2_f,psi2_f-psi_std2_f,psi2_f+psi_std2_f,alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle (deg)')
    plt.xlim(80,160)
    plt.ylim(-5, 5)
    # plt.title('Course Angle over Episode')
    # plt.legend(['Course angle', 'Target course angle'])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('1_course_angle.png')
    # plt.show()
    plt.close()

    # violin plot of course angle after 1/2 way pt
    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)

    labels = ['PPO Policy','PID']

    fig,ax = plt.subplots()
    ax.set_xlabel('Label')
    ax.set_ylabel('Course Angle (deg)')

    parts = ax.violinplot([psi2,psi2_f],showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('forestgreen')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    set_axis_style(ax,labels)

    plt.savefig('1_course_angle_violin.png')
    # plt.show()
    plt.close()

def plotCourseAngle_PID(simData_f,simTime_f,target_course):
    """
    plotCourseAngle(simData,simTime,simData_fixed,simTime_fixed,target_course) takes in the simData 
    and simTime for the PPO policy as well as the fixed PID controller, and target course angle, 
    and plots the course angle over time.
    """

    # Course angle plots over time - fixed PID policy
    psi_f = R2D(np.average(ssa(simData_f),axis=1,keepdims=True))
    psi_std_f = R2D(np.std(ssa(simData_f),axis=1,keepdims=True))
    half_f = len(psi_f)//2
    psi1_f = psi_f[0:half_f].T[0]
    psi2_f = psi_f[half_f:].T[0]
    psi_std1_f = psi_std_f[0:half_f].T[0]
    psi_std2_f = psi_std_f[half_f:].T[0]
    time1_f = simTime_f[0:half_f]
    time2_f = simTime_f[half_f:]
    psi_d1 = np.full(len(psi1_f),target_course)
    psi_d2 = np.full(len(psi2_f),target_course)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(time1_f,psi1_f)
    plt.plot(time1_f,psi_d1,'--k')
    plt.fill_between(time1_f,psi1_f-psi_std1_f,psi1_f+psi_std1_f,alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle (deg)')
    plt.xlim(0,80)
    plt.ylim(-45, 180)
    # plt.title('Course Angle over Episode')
    plt.legend(['PPO','PID','Target course angle'])
    plt.grid(True)

    plt.subplot(212)
    plt.plot(time2_f,psi2_f)
    plt.plot(time2_f,psi_d2,'--k')
    plt.fill_between(time2_f,psi2_f-psi_std2_f,psi2_f+psi_std2_f,alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle (deg)')
    plt.xlim(80,160)
    plt.ylim(-5, 5)
    # plt.title('Course Angle over Episode')
    # plt.legend(['Course angle', 'Target course angle'])
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plotActionsCourseAngle(kp,ki,kd,simTime):
    """
    plotActionsCourseAngle(kp,ki,kd,simTime) takes in the actions over time 
    for the course angle evaluation, and the simTime, and plots the max actions.
    """

    # get max action values over time for kp, ki, kd
    kp_avg = np.average(kp,axis=1,keepdims=True)
    ki_avg = np.average(ki,axis=1,keepdims=True)
    kd_avg = np.average(kd,axis=1,keepdims=True)

    plt.figure(3)
    plt.plot(simTime,kp_avg)
    plt.plot(simTime,ki_avg)
    plt.plot(simTime,kd_avg)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['kp','ki','kd'])
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Action Value')
    plt.title('Course Angle Evaluation Action Values')

    plt.savefig('1_crs_angle_actions.png')
    # plt.show()
    plt.close()


def plotTraj(simData,simTime,simData_f,simTime_f,path_x,path_y,pref,suf):
    """
    plotTraj(simData,simTime,simData_f,simTime_f,path_x,path_y,pref,suf) takes in the 
    simData for the trajectory evaluation, the simTime, and the desired path x and y 
    coordinates and plots the X-Y position, the course angle error, the trajectory error, 
    and a violin plot of the course angle error for the PPO policy vs. the fixed PID controller.
    """
    
    # USV position-----------------------------------------------------------------
    x = simData[:,0]
    y = simData[:,1]
    x_f = simData_f[:,0]
    y_f = simData_f[:,1]

    # X, Y position plot - PPO
    plt.figure(1)
    plt.plot(x,y)
    plt.plot(path_x,path_y,'--k')
    plt.xlabel('X / East (m)')
    plt.ylabel('Y / North (m)')
    # plt.title('X-Y Trajectory')
    plt.grid(True)
    plt.legend(['PPO-PID','Target Path'])
    # plt.xlim(-600,600)
    # plt.ylim(-600,600)

    plt.savefig(pref + '_traj_PPO_' + suf + '.png')
    # plt.show()
    plt.close()

    # X, Y position plot - PID
    plt.figure(2)
    plt.plot(x_f,y_f,'g')
    plt.plot(path_x,path_y,'--k')
    plt.xlabel('X / East (m)')
    plt.ylabel('Y / North (m)')
    # plt.title('X-Y Trajectory')
    plt.grid(True)
    plt.legend(['PID','Target Path'])
    # plt.xlim(-600,600)
    # plt.ylim(-600,600)

    plt.savefig(pref + '_traj_PID_' + suf + '.png')
    # plt.show()
    plt.close()

    # Course angle error--------------------------------------------------------
    yaw_err = simData[:,17] * 180/math.pi
    yaw_err_f = simData_f[:,17] * 180/math.pi

    # PPO
    plt.figure(3)
    plt.plot(simTime,yaw_err)
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle Error (deg)')
    # plt.title('Course Angle Error')
    plt.grid(True)
    plt.legend(['PPO-PID'])
    # plt.xlim(-600,600)
    # plt.ylim(-600,600)

    plt.savefig(pref + '_PPO_course_angle_err_' + suf + '.png')
    # plt.show()
    plt.close()

    # PID
    plt.figure(4)
    plt.plot(simTime_f,yaw_err_f,'g')
    plt.xlabel('Time (s)')
    plt.ylabel('Course Angle Error (deg)')
    # plt.title('Course Angle Error')
    plt.grid(True)
    plt.legend(['PID'])
    # plt.xlim(-600,600)
    # plt.ylim(-600,600)

    plt.savefig(pref + '_PID_course_angle_err_' + suf + '.png')
    # plt.show()
    plt.close()

    # Path error----------------------------------------------------------
    path_err = simData[:,18]
    path_err_f = simData_f[:,18]

    # PPO
    plt.figure(5)
    plt.plot(simTime,path_err)
    plt.xlabel('Time (s)')
    plt.ylabel('Trajectory Error (m)')
    # plt.title('Trajectory Error')
    plt.grid(True)
    plt.legend(['PPO-PID'])
    # plt.xlim(-600,600)
    # plt.ylim(-600,600)

    plt.savefig(pref + '_PPO_path_err_' + suf + '.png')
    # plt.show()
    plt.close()

    # PID
    plt.figure(6)
    plt.plot(simTime_f,path_err_f,'g')
    plt.xlabel('Time (s)')
    plt.ylabel('Trajectory Error (m)')
    # plt.title('Trajectory Error')
    plt.grid(True)
    plt.legend(['PID'])
    # plt.xlim(-600,600)
    # plt.ylim(-600,600)

    plt.savefig(pref + '_PID_path_err_' + suf + '.png')
    # plt.show()
    plt.close()


    # violin plot of course angle error--------------------------------------
    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)

    labels = ['PPO Policy','PID']

    fig4,ax4 = plt.subplots()
    ax4.set_xlabel('Label')
    ax4.set_ylabel('Course Angle Error (deg)')
    ax4.set_ylim([-15,15])

    parts = ax4.violinplot([yaw_err,yaw_err_f],showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('royalblue')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    set_axis_style(ax4,labels)

    plt.savefig(pref + '_course_angle_err_violin_' + suf + '.png')
    # plt.show()
    plt.close()


def plotStates(simData,simTime,pref,suf):
    """
    plotStates(simData,simTime) takes in the simData for the trajectory 
    evaluation, and the simTime, and plots the vehicle states.
    """

    # states
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))
    psi = R2D(ssa(simData[:, 5]))
    r = R2D(simData[:, 11])

    plt.figure(5,figsize=(10,5))
    plt.suptitle('USV States')

    # speed
    plt.subplot(221)
    plt.plot(simTime,U)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Speed (m/s)'])
    plt.grid(True)

    # course angle
    plt.subplot(222)
    plt.plot(simTime,psi)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Course Angle (deg)'])
    plt.grid(True)

    # surge/sway velocity
    plt.subplot(223)
    plt.plot(simTime,u)
    plt.plot(simTime,v)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Surge Vel (m/s)','Sway Vel (m/s)'])
    plt.xlabel('Time (s)')
    plt.grid(True)

    # yaw rate
    plt.subplot(224)
    plt.plot(simTime,r)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Yaw Rate (deg/s)'])
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(pref + '_usv_states_' + suf + '.png')
    # plt.show()
    plt.close()


def plotControls(simData,simTime,pref,suf):
    """
    plotControls(simData,simTime) takes in the simData for the trajectory 
    evaluation, and the simTime, and plots the vehicle control input vs. 
    actual.
    """

    # Controls
    u_control1 = simData[:,12]
    u_control2 = simData[:,13]
    u_actual1 = simData[:,14]
    u_actual2 = simData[:,15]
    tau_X = simData[:,19]
    tau_N = simData[:,20]

    plt.figure(6,figsize=(10,5))
    plt.suptitle('USV Control')

    # left propeller command
    plt.subplot(221)
    plt.plot(simTime,u_control1)
    plt.plot(simTime,u_actual1)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Left Prop Rev Cmd (rpm)','Left Prop Rev Actual (rpm)'])
    plt.grid(True)

    # surge control force
    plt.subplot(222)
    plt.plot(simTime,tau_X)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Surge Control Force (N)'])
    plt.grid(True)

    # right propeller command
    plt.subplot(223)
    plt.plot(simTime,u_control2)
    plt.plot(simTime,u_actual2)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Right Prop Rev Cmd (rpm)','Right Prop Rev Actual (rpm)'])
    plt.grid(True)
    plt.xlabel('Time (s)')

    # Yaw Control Moment
    plt.subplot(224)
    plt.plot(simTime,tau_N)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['Yaw Control Moment (Nm)'])
    plt.grid(True)
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(pref + '_usv_controls_' + suf + '.png')
    # plt.show()
    plt.close()


def plotActionsTraj(action,simTime,pref,suf):
    """
    plotActionsTraj(action,simTime) takes in the actions over time 
    for the trajectory evaluation, and the simTime, and plots the actions.
    """

    # breakout action values
    kp = action[:,0]
    ki = action[:,1]
    kd = action[:,2]

    plt.figure(7)
    plt.plot(simTime[1:],kp)
    plt.plot(simTime[1:],ki)
    plt.plot(simTime[1:],kd)
    # plt.xlim(0,90)
    # plt.ylim(-45, 180)
    plt.legend(['kp','ki','kd'])
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Action Value')
    plt.title('Trajectory Evaluation Action Values')

    plt.savefig(pref + '_traj_actions_' + suf + '.png')
    # plt.show()
    plt.close()