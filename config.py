import torch
import math
import numpy as np

config = {
    "num_envs": 32,
    "device": "cpu",
    "seed": 0,
    "sim_dt": 0.05,
    "policy_dt": 0.05,

    "sim_max_time": 160,
    "max_steps": 5_000_000_000,
    "policy_cls": "MlpPolicy",
    "policy_kwargs": dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[32, 32],
    ),
    "verbose": 1,
    "rollout_steps": 128,
    "minibatch_size": 32,
    "model_save_freq": 1_000_000, 

    # observation space limits
    "vel_lim": 3.1,
    "acc_lim": 2.5,
    "ang_vel_lim": 10*math.pi,
    "ang_acc_lim": 0.20,
    "yaw_err_lim": math.pi,
    "prop_rpm_max": 103.93,
    "prop_rpm_min": -101.74,
    "delTn_lim": 350,
    "P_limit": 2*math.radians(180),
    "I_limit": math.radians(180),
    "D_limit": 4*math.radians(180),

    # PID coefficient limits
    "Kp_limit": 4000,
    "Ki_limit": 300,
    "Kd_limit": 300,

    # reward and truncate limits
    "reward_alpha_coefficient": 0.078125,
    "reward_beta_coefficient": -0.1,
    "reward_gamma_coefficient": 0.0,
    "prop_act_penalty_lim": 40,

    # reset limits
    "tau_X": 120.0,
    "water_curr_vel_low": 0.0,
    "water_curr_vel_high": 1.0,
    "water_curr_angle_low": math.radians(-180),
    "water_curr_angle_high": math.radians(180),

    # Desired course angle
    "target_course_angle": math.radians(0),
    "angle_error_lim": 0.09,
    "target_hold_time": 2,

    # Varying current
    "mu": 0.15,
    "Vmin": 0.10,
}