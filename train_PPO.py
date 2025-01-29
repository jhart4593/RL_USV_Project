from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import traceback

from config import config 
from env import USVEnv
import rewards

run = wandb.init(
    project="USV_env",
    config=config,
    sync_tensorboard=True,
    # monitor_gym=True,
    save_code=True,
)

# Save checkpoint every model_save_freq steps
save_freq = config["model_save_freq"]
checkpoint_callback = CheckpointCallback(
    save_freq = max(save_freq // config["num_envs"], 1),
    save_path="./model_logs/",
    name_prefix="rl_model"
)

# Save config, env, and rewards file for each training run to wandb
wandb.save( "./config.py")
wandb.save("rewards.py")
wandb.save("./env.py")

env = make_vec_env(USVEnv,n_envs=config["num_envs"])

model = PPO(
    config["policy_cls"],
    env,
    policy_kwargs=config["policy_kwargs"],
    verbose=config["verbose"],
    device=config["device"],
    tensorboard_log=f"runs/{run.id}",
    n_steps=config["rollout_steps"],
    batch_size=config["minibatch_size"]
)
try:
    model.learn(
        total_timesteps=config["max_steps"],
        callback=[
            WandbCallback(
                verbose=config["verbose"],
                model_save_path=None,  # f"models/{run.id}"
                model_save_freq=0,  # 100
                gradient_save_freq=0,  # 100
            ),

            checkpoint_callback
        ],
    )
    save_dir = "./final_model"
    model.save(save_dir + "PPO_USV")

except:
    traceback.print_exc()
run.finish()