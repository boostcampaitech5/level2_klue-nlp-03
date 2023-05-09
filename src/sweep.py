import wandb
import yaml
from train import train

def main():
    with wandb.init() as run:
        cfg=wandb.config
        train(cfg)

if __name__=="__main__":
    with open("./sweep.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(
        sweep=cfg, 
        entity="ggul_tiger",
        project='KLUE',
    )

    wandb.agent(
        sweep_id, 
        function=main, 
        entity="ggul_tiger",
        project='KLUE',
        count=10
    )
