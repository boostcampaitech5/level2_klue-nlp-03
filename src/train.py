import yaml
import pytorch_lightning as pl

from models import BaseModel
from loader import KLUEDataLoader
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import get_result_name

# load config
with open("./config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# set random seed
pl.seed_everything(cfg["seed"])

# get result name
result_name = get_result_name()

tokenizer = AutoTokenizer.from_pretrained(
    cfg["train"]["model_name"], model_max_length=cfg["train"]["max_len"]
)
model = BaseModel(tokenizer, cfg)
dataloader = KLUEDataLoader(tokenizer, cfg)

# wandb logger, ggul_tiger 팀으로 run이 기록됩니다.
logger = WandbLogger(name=result_name, project="KLUE", entity="ggul_tiger")
logger.experiment.config.update(cfg)

trainer = pl.Trainer(
    accelerator="gpu",
    strategy="ddp",
    max_epochs=cfg["train"]["epoch"],
    logger=logger,
    default_root_dir=cfg["dir"]["result_dir"] + result_name,
    log_every_n_steps=50,
    callbacks=[
        ModelCheckpoint(
            dirpath=cfg["dir"]["result_dir"] + result_name,
            filename="best_model",
            monitor="val_loss",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg["train"]["patience"],
            verbose=True,
        ),
    ],
)

trainer.fit(model=model, datamodule=dataloader)
