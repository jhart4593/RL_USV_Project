# Reinforcement Learning for Adaptive Control of an Unmanned Surface Vehicle

## Project Summary
Unmanned Surface Vehicles (USVs) are being used across an array of industries, the automation of which contributes to their appeal. A critical component of their automation is the course keeping task – the ability to adjust its heading angle to the desired heading and hold it. One method that can be applied to execute this task is a reinforcement learning (RL) - PID controller, which combines the advantages of the intuitiveness of a PID controller, and the robustness of RL in dynamic environments, while minimizing the difficulty in parameter tuning of a PID controller, and the black-box nature of RL. An adaptive RL-PID controller is developed, utilizing the proximal policy optimization RL algorithm. Its performance is verified in both course keeping and path following tasks and accomplishes these tasks as well and slightly better than a fixed PID controller. More work should be done to fine tune the RL training, and to conduct the training on higher end hardware to produce a policy that greatly exceeds the performance of a fixed PID controller. 

## Requirements
Reference 'requirements' text file for necessary packages to install for the code to run.

## Code Organization
All code for training of the RL agent is within the 'training' folder. The code for evaluation of the learned RL policy is located within the 'evaluation' folder. The 'utilities' folder houses all of the supporting Python scripts to include the USV simulator, reward function, and plotting utilities.

## Running the Code
1. Run train_PPO script in 'training' folder to train an RL agent. Change parameters of the training in the config file within the same folder.
2. Run the evaluate script within the 'evaluation' folder to generate plots comparing a fixed PID controller to the RL-PID controller. Parameters for the evaluation are adjusted in the eval_config file in the same folder.
