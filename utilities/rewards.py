import math
import numpy as np
from training.config import config as cfg

# Rewards function
def get_rewards(yaw_err, simData):
    """
    r_t = get_rewards(yaw_err, pitch_err, inc_r, inc_s, step) returns the RL reward value
    using arrays of current and all past angular errors, incremental rudder angle commands, 
    and the current step number within the current episode.
    """

    # break out values needed from input arrays
    e_yaw = yaw_err[-1]; 
    yaw_err_list = yaw_err[2:]
    e_yaw_min = np.min(np.absolute(yaw_err_list))
    time = simData[-1][16]
    
    alpha = cfg["reward_alpha_coefficient"]
    beta = cfg["reward_beta_coefficient"]
    gamma = cfg["reward_gamma_coefficient"]

    # define functions for T operator of reward function - reward small course angle error, 
    # penalize if course angle error grows
    def T_op(err, err_min):
        if abs(err) <= (abs(err_min) + 0.001):
            T = math.exp(-abs(err))
            # T = T / (simData.shape[0] + 1)
            # T = T * (cfg["sim_max_time"] - time)/cfg["sim_max_time"]
        else:
            T = -abs(err) / math.pi
        return T
    
    # Initialize prop command penalty
    neg_prop_act = 0
    
    # Penalizing if greatest difference in last 5 prop commands is over threshold in config
    # prop_cmd_lst1 = simData[:,12]
    # prop_cmd_lst2 = simData[:,13]
    # pen_lim = cfg["prop_act_penalty_lim"]
    # max_diff = cfg["prop_rpm_max"] - cfg["prop_rpm_min"]
    # lg1 = len(prop_cmd_lst1)
    # lg2 = len(prop_cmd_lst2)
    # if lg1 < 6:
    #     end1 = prop_cmd_lst1
    #     end2 = prop_cmd_lst2
    # else:
    #     end1 = prop_cmd_lst1[lg1-5:lg1]
    #     end2 = prop_cmd_lst2[lg2-5:lg2]
    
    # diff1 = abs(max(end1) - min(end1))
    # diff2 = abs(max(end2) - min(end2))

    # if (diff1 >= pen_lim) or (diff2 >= pen_lim):
    #     diff = max(diff1,diff2)
    #     exp_val = 1.5*(1-(diff - pen_lim)/(max_diff - pen_lim))
    #     neg_prop_act += beta * math.exp(-abs(exp_val))

    # Calculate individual reward terms
    T_yaw = alpha * T_op(e_yaw, e_yaw_min) 

    time_pen = gamma * time / cfg["sim_max_time"]

    # Output individual reward terms as well as sum
    indiv_terms = [T_yaw,neg_prop_act,time_pen]

    r_t = (T_yaw + neg_prop_act + time_pen)

    return r_t, indiv_terms


# test the function
if __name__ == "__main__":

    yaw_err = [3,6,2]
    one = np.array([1,2,3,4,5,6]+[7,8,9,10,11,12]+[50,100]+[45,90]+[2.4])
    two = np.array([6,5,4,3,2,1]+[12,11,10,9,8,7]+[90,90]+[50,100]+[2.6])
    data = np.vstack((one,two))

    [r_t,indiv_r_terms] = get_rewards(yaw_err,data)

    print(r_t)
    print(indiv_r_terms)