from typing import Any, Callable, Dict, Iterable, List,  Optional, Type

import numpy as np
import math
import wandb
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import config
from utilities.rewards import get_rewards
from utilities.otter import otter
from utilities.utils import get_obs, get_obs_norm, PID, get_current
from utilities.fossen_gnc import ssa


class USVEnv(gym.Env):
    """
    Custom Environment that follows gym interface for USV adaptive PID controller.
    """
    metadata = {
        "render_modes": ["human"]
        }

    def __init__(
        self,
        render_mode = None,
        config: Dict = config,
        get_rewards: Callable = get_rewards,
        get_obs: Callable = get_obs,
        get_obs_norm: Callable = get_obs_norm,
        PID: Callable = PID,
        otter: Callable = otter,
        get_current: Callable = get_current,
        ):

        super(USVEnv, self).__init__()

        self.render_mode = render_mode
        self.cfg = config
        self.get_rewards = get_rewards
        self.get_obs = get_obs
        self.get_obs_norm = get_obs_norm
        self.PID = PID
        self.otter = otter
        self.get_current = get_current

        # initialize anything necessary for environment
        self.num_term = 0
        self.num_trunc = 0
        self.simData = np.empty([0,17],float)
        self.t = 0
        self.sampleTime = self.cfg["sim_dt"]
        self.max_time = self.cfg["sim_max_time"]
        self.tau_X = self.cfg["tau_X"]  # surge force constant input
        self.Vc = 0
        self.beta_c = 0
        self.vehicle = self.otter(tau_X=self.tau_X)

        self.yaw_err = [0,0]; self.tau_N = 0

        self.eta = np.zeros(6)
        self.nu = np.zeros(6)
        self.nu_dot = np.zeros(6)
        self.u_actual = np.zeros(2)
        self.u_control = np.zeros(2)

        self.kp_high = self.cfg["Kp_limit"]
        self.ki_high = self.cfg["Ki_limit"]
        self.kd_high = self.cfg["Kd_limit"]

        self.target_course = self.cfg["target_course_angle"]

        # Define action and observation space
        # action space is limits on PID coefficients - Kp, Ki, Kd for PID heading controller
        # normalized action space
        self.action_space = spaces.Box(
            low=np.array([-1,-1,-1]),
            high=np.array([1,1,1]),
            dtype=np.float32
            )

        # The observation space is the state vector - [yaw_err, [u,v,r], [udot,vdot,rdot],
        # [n1,n2], delTn, PID terms] where the dot terms are the acceleration terms in surge, sway, and 
        # yaw, n1 and n2 are the propeller rpms, delTn is the incremental PID controller output,
        # and the PID terms are error in yaw, error in yaw rate, and the error integral term for the PID controller

        # yaw_err_lim = self.cfg["yaw_err_lim"]
        # vel_lim = self.cfg["vel_lim"]
        # acc_lim = self.cfg["acc_lim"]
        # ang_vel_lim = self.cfg["ang_vel_lim"]
        # ang_acc_lim = self.cfg["ang_acc_lim"]
        # prop_rpm_max = self.cfg["prop_rpm_max"]
        # prop_rpm_min = self.cfg["prop_rpm_min"]
        # delTn_lim = self.cfg["delTn_lim"]
        # P_lim = self.cfg["P_limit"]
        # I_lim = self.cfg["I_limit"]
        # D_lim = self.cfg["D_limit"]

        # min_vel = [-vel_lim, -vel_lim, -ang_vel_lim]
        # max_vel = [vel_lim, vel_lim, ang_vel_lim]        

        # min_acc = [-acc_lim, -acc_lim, -ang_acc_lim]
        # max_acc = [acc_lim, acc_lim, ang_acc_lim]

        # low_vec = ([-yaw_err_lim] + min_vel + min_acc + [prop_rpm_min, prop_rpm_min] + [-delTn_lim] + [-P_lim,-I_lim,-D_lim])
        # high_vec = ([yaw_err_lim] + max_vel + max_acc + [prop_rpm_max, prop_rpm_max] + [delTn_lim] + [P_lim,I_lim,D_lim])

        # self.observation_space = spaces.Box(low=np.array(low_vec), high=
        #                                     np.array(high_vec), dtype=np.float32)
        
        # Using normalized observation space from -1 to 1
        self.observation_space = spaces.Box(
            low=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
            high=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1]),
            dtype=np.float32
            )

        # Initialize plotting variables for tracking
        self.plot = None
        self.counter = 0

        # Reward sum for each timestep, total episode reward
        self.reward_sum = 0
        self.episode_reward = 0
        self.yaw_e = 0
        self.prop_act = 0
        self.time_pen = 0
        self.yaw_e_ep = 0
        self.prop_act_ep = 0
        self.time_pen_ep = 0
        
        # terminate criteria counter --> holds target course for x seconds
        self.course_hold = 0


    def reset(self, seed=None, options=None):
        # called to initiate a new episode, called before step function
        # also called whenever terminated or truncated is issued
        # resets environment to an initial state

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Render the data for episode about to be reset and reset counter
        if (self.counter % 10 == 0) and (self.t > 0): 
            plot_1, plot_2 = self.render()
            wandb.log({"plot/X_Y": wandb.Image(plot_1)})
            wandb.log({"plot/Course_angle": wandb.Image(plot_2)})
            plt.close('all')
            
        self.counter += 1

        # Reset time counter and course hold counter
        self.t = 0
        self.course_hold = 0

        # Track cumulative episode rewards in wandb and reset reward sum values
        self.episode_reward = self.reward_sum
        self.yaw_e_ep = self.yaw_e
        self.prop_act_ep = self.prop_act
        self.time_pen_ep = self.time_pen

        self.reward_sum = 0
        self.yaw_e = 0
        self.prop_act = 0
        self.time_pen = 0

        # set constant surge control input
        self.tau_X = self.cfg["tau_X"]

        # randomly sample water current velocity and angle over range in config file
        self.Vc = self.np_random.uniform(low=self.cfg["water_curr_vel_low"], high=self.cfg["water_curr_vel_high"])
        self.beta_c = self.np_random.uniform(low=self.cfg["water_curr_angle_low"], high=self.cfg["water_curr_angle_high"])
        [self.current_mag,self.current_angle] = self.get_current(self.Vc,self.cfg["mu"],self.cfg["water_curr_vel_high"],self.cfg["Vmin"],self.beta_c)

        # Reset vehicle instance
        self.vehicle = self.otter(tau_X=self.tau_X)

        # set initital state of AUV
        init_course_angle = self.np_random.uniform(low=math.radians(-180), high=math.radians(180))
        self.eta = np.array([0,0,0,0,0,init_course_angle])
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) 
        self.nu_dot = np.array([0, 0, 0, 0, 0, 0], float)
        self.u_actual = np.array([0, 0], float)    
        self.u_control = np.array([0, 0], float)  

        # store simulation data
        self.simData = np.array(list(self.eta) + list(self.nu) + list(self.u_control) + list(self.u_actual) + [self.t])

        # reset other values
        self.yaw_err = [0,0,ssa(self.target_course - self.eta[5])]
        self.tau_N = 0

        # determine observation from initial state
        observation = self.get_obs_norm(self.yaw_err, self.nu, self.nu_dot, self.u_actual, 0, [0,0,0])

        # determine info to return
        info = {}

        # if self.render_mode == "human":
        #   self.render()

        return observation, info


    def step(self, action):
        # using agent actions, run one timestep of the environment's dynamics
        # don't initiate disturbance current until time > 1s
        if self.t < 1:
            Vc = 0
        else:
            Vc = self.current_mag[self.counter+1][0]

        beta_c = self.current_angle[self.counter+1]

        # handling normalized action space
        def kp_norm(norm_action):
            kp = (norm_action + 1)/2 * self.kp_high
            return kp
        
        def ki_norm(norm_action):
            ki = (norm_action + 1)/2 * self.ki_high
            return ki
        
        def kd_norm(norm_action):
            kd = (norm_action + 1)/2 * self.kd_high
            return kd
        
        def inc_act(act,base_val):
            inc = base_val * 0.8 * act
            return base_val + inc
        
        # norm_action = np.array([kp_norm(action[0]),ki_norm(action[1]),kd_norm(action[2])])
        # uses +/- 80% of tuned PID coefficients
        norm_action = np.array([inc_act(action[0],414.0),inc_act(action[1],0.001),inc_act(action[2],50.0)])
        
        [self.tau_N,del_tau_N,cont_coeffs] = (
            self.PID(norm_action,self.tau_N,self.yaw_err)
        )

        self.u_control = self.vehicle.controlAllocation(self.tau_X,self.tau_N)

        [self.eta,self.nu,self.u_actual,self.nu_dot,_] = (
            self.vehicle.dynamics(self.eta,self.nu,self.u_actual,self.u_control,self.sampleTime,Vc,beta_c)
        )

        # Add to yaw error list, iterate course hold counter if holding
        self.yaw_err.append(ssa(self.target_course - self.eta[5]))

        if abs(self.yaw_err[-1]) <= self.cfg["angle_error_lim"]:
            self.course_hold += 1
        else:
            self.course_hold = 0

        # determine updated observation based on action taken
        observation = self.get_obs_norm(self.yaw_err, self.nu, self.nu_dot, self.u_actual, del_tau_N, cont_coeffs)

        # Iterate time counter and plotting counter
        self.t += self.sampleTime
        # self.counter += 1

        # Store simulation data
        self.simData = np.vstack((self.simData,np.array(list(self.eta) + list(self.nu) + list(self.u_control) + list(self.u_actual) + [self.t])))

        # calculate reward, log on wandb
        [reward,indiv_rew_terms] = self.get_rewards(self.yaw_err,self.simData)

        self.reward_sum += reward
        self.yaw_e += indiv_rew_terms[0]
        self.prop_act += indiv_rew_terms[1]
        self.time_pen += indiv_rew_terms[2]

        wandb.log({"train/reward":self.reward_sum})
        wandb.log({"train/episode_return":self.episode_reward})
        wandb.log({"reward/yaw_err":self.yaw_e_ep})
        wandb.log({"reward/prop_act":self.prop_act_ep})
        wandb.log({"reward/time_pen":self.time_pen_ep})
        
        # set terminated criteria - if reached desired course angle or hold course angle for desired amount of time
        terminated = False
        # if abs(self.yaw_err[-1]) <= self.cfg["angle_error_lim"]:
        #     terminated = True

        if (self.t >= self.max_time):
           terminated = True

        # if self.course_hold >= (self.cfg["target_hold_time"]/self.cfg["sim_dt"]):
        #     terminated = True

        # set truncated criteria
        # if over max time, if roll/pitch over prescribed limits
        truncated = False
        if (self.t > (self.max_time + 10)):
           truncated = True

        # set value based on terminated or truncated for episode 
        if terminated:
            self.num_term += 1

        if truncated:
            self.num_trunc += 1 
           
        # determine info to return
        info = {}
        
        # Wandb logging for other parameters
        wandb.log({"state/prop1_rpm":self.u_actual[0]})
        wandb.log({"state/prop2_rpm":self.u_actual[1]})
        wandb.log({"state/tau_N":self.tau_N})
        wandb.log({"state/delta_tau_N":del_tau_N})
        wandb.log({"state/yaw_error":self.yaw_err[-1]})
        wandb.log({"state/Kp":norm_action[0]})
        wandb.log({"state/Ki":norm_action[1]})
        wandb.log({"state/Kd":norm_action[2]})

        # log number of episodes that are terminated vs truncated
        wandb.log({"reward/num_term":self.num_term})
        wandb.log({"reward/num_trunc":self.num_trunc})


        return observation, reward, terminated, truncated, info


    def render(self):

      # State vectors
      x = self.simData[:,0]
      y = self.simData[:,1]
      psi = ssa(self.simData[:,5]) * 180/math.pi
      time = self.simData[:,16]
      psi_d = np.full(len(psi),self.target_course)

      # X, Y position plot
      fig1, ax1 = plt.subplots()
      ax1.plot(x,y)
      ax1.set_xlabel('X / East (m)')
      ax1.set_ylabel('Y / North (m)')
      ax1.set_title('X-Y Trajectory')
      ax1.grid(True)

      # Course angle plot
      fig2, ax2 = plt.subplots()
      ax2.plot(time,psi)
      ax2.plot(time,psi_d,'--k')
      ax2.set_xlabel('Time (s)')
      ax2.set_ylabel('Course Angle (deg)')
      ax2.set_ylim(-180, 180)
      ax2.set_title('Course Angle over Episode')
      ax2.legend(['Course angle', 'Desired course angle'])
      ax2.grid(True)

      return fig1, fig2


    def close(self):
      return
    

# Perform environment checks
if __name__ == "__main__":
    import time
    from stable_baselines3.common.env_checker import check_env

    # SB3 environment check
    # If the environment don't follow the correct interface, an error will be thrown
    env = USVEnv()
    
    check_env(env, warn=True)

    # test environment
    obs = env.reset()
    # env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    T = 100
    now = time.time()
    for _ in range(T):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
    print(f"{int(T/(time.time() - now)):_d} steps/second")
    env.render()
    plt.show()