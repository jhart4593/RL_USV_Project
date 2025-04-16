import gymnasium as gym

from eval_env import USVEnv_eval 
from eval_env_LOS import USVEnv_eval_LOS
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import numpy as np
import math
from training.config import config
from eval_config import eval_config
from utilities.plot_utils import plotCourseAngle, plotTraj, plotControls, plotStates, plotActionsTraj, plotActionsCourseAngle, plotCourseAngle_PID
from utilities.otter_sim import sim_otter_course_angle, sim_otter_LOS

policy = "./model_logs/rl_model_38000000_steps.zip"

# Initial course angle evaluation -----------------------------------------------------------------------------------
env = make_vec_env(USVEnv_eval)
model = PPO.load(policy, env=env)
num_steps = eval_config["eval_steps"]
disturbance_angle = eval_config["water_curr_angle"]

simData_total = np.empty([num_steps+1,0],float)
kp_total = np.empty([num_steps+1,0],float)
ki_total = np.empty([num_steps+1,0],float)
kd_total = np.empty([num_steps+1,0],float)

for kk in range(len(disturbance_angle)):
    obs = env.reset()
    # env.set_attr("beta_c",disturbance_angle[kk])
    env.env_method("set_beta_c",disturbance_angle[kk])
    print(env.env_method("get_wrapper_attr", "beta_c"))
    info = env.reset_infos
    # print(info[0]["Data"])
    simData = np.array([info[0]["Data"][0]])
    kp = np.array([0]); ki = np.array([0]); kd = np.array([0])

    for ii in range(num_steps):
        t = ii * config["sim_dt"]
        action, _ = model.predict(obs,deterministic=True)
        obs, rew, done, info = env.step(action)

        # print(info[0]["Data"])
        simData= np.vstack((simData,np.array([info[0]["Data"][0]])))

        # if t%10 == 0:
        #     print(action)
        # save action values
        kp = np.vstack((kp,action[0][0]))
        ki = np.vstack((ki,action[0][1]))
        kd = np.vstack((kd,action[0][2]))

        if done:
            break
    
    print(info[0]["Data"][1])
    simData_total = np.hstack((simData_total,simData))
    kp_total = np.hstack((kp_total,kp))
    ki_total = np.hstack((ki_total,ki))
    kd_total = np.hstack((kd_total,kd))

simTime = []
for jj in range(simData.shape[0]):
    t = jj * config["sim_dt"]
    simTime.append(t)

# print(simData)
# print(simTime)
# print(simData_total[1800,:])

# Fixed PID data
simData_total_fixed, simTime_fixed = sim_otter_course_angle()

# Plots
plotCourseAngle(simData_total,simTime,simData_total_fixed,simTime_fixed,eval_config["target_course_angle"])
plotActionsCourseAngle(kp_total,ki_total,kd_total,simTime)

# plotCourseAngle_PID(simData_total_fixed,simTime_fixed,eval_config["target_course_angle"])


# LOS Evaluation - circle trajectory ----------------------------------------------------------------------------
env = make_vec_env(USVEnv_eval_LOS)
model = PPO.load(policy, env=env)
num_steps = eval_config["eval_steps_LOS"]

obs = env.reset()
info = env.reset_infos
act = np.empty([0,3],float)

for ii in range(num_steps):
    t = ii * config["sim_dt"]
    action, _ = model.predict(obs,deterministic=True)
    obs, rew, done, info = env.step(action)

    # if t%10 == 0:
    #     print(action)

    # save action values
    act = np.vstack((act,np.array(action)))

    if done:
        break

# print(info[0]["Data"])
simData = info[0]["Data"]

simTime = []
for jj in range(simData.shape[0]):
    t = jj * config["sim_dt"]
    simTime.append(t)

# fixed PID trajectory data
simTime_f, simData_f = sim_otter_LOS(eval_config["path_x"],eval_config["path_y"])


# Plots
plotTraj(simData,simTime,simData_f,simTime_f,eval_config["path_x"],eval_config["path_y"],'2','')
plotStates(simData,simTime,'2','')
plotStates(simData_f,simTime_f,'2','PID')
plotControls(simData,simTime,'2','')
plotControls(simData_f,simTime_f,'2','PID')
plotActionsTraj(act,simTime,'2','')



# LOS Evaluation - M trajectory -------------------------------------------------------------------------------
env = make_vec_env(USVEnv_eval_LOS)
model = PPO.load(policy, env=env)
num_steps = eval_config["eval_steps_LOS"]

obs = env.reset()
M_path = np.vstack((np.array(eval_config["path_x1"]),np.array(eval_config["path_y1"])))
env.env_method("set_path",M_path)
info = env.reset_infos
act = np.empty([0,3],float)

for ii in range(num_steps):
    t = ii * config["sim_dt"]
    action, _ = model.predict(obs,deterministic=True)
    obs, rew, done, info = env.step(action)

    # if t%10 == 0:
    #     print(action)

    # save action values
    act = np.vstack((act,np.array(action)))

    if done:
        break

# print(info[0]["Data"])
simData = info[0]["Data"]

simTime = []
for jj in range(simData.shape[0]):
    t = jj * config["sim_dt"]
    simTime.append(t)

# fixed PID trajectory data
simTime_f, simData_f = sim_otter_LOS(eval_config["path_x1"],eval_config["path_y1"])

# Plots
plotTraj(simData,simTime,simData_f,simTime_f,eval_config["path_x1"],eval_config["path_y1"],'3','M')
plotStates(simData,simTime,'3','M')
plotStates(simData_f,simTime_f,'3','M_PID')
plotControls(simData,simTime,'3','M')
plotControls(simData_f,simTime_f,'3','M_PID')
plotActionsTraj(act,simTime,'3','M')