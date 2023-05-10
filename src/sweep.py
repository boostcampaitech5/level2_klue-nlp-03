import wandb
import yaml
from train import train
from utils import get_result_name

def main():
    result_name = get_result_name()
    with wandb.init(name=result_name) as run:
        cfg=wandb.config
        train(cfg,result_name)

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
        count=20
    )