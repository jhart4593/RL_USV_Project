# https://docs.wandb.ai/ref/python/public-api/api/
# https://docs.wandb.ai/guides/track/public-api-guide/

import wandb

path = "L4DSC_project/USV_env/lckrye5e"

api = wandb.Api()
run = api.run(path)
run.file("rewards.py").download(root="./wandb")