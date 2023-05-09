import yaml
import pytorch_lightning as pl
import pandas as pd

from models import BaseModel
from loader import KLUEDataLoader
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import get_result_name, num_to_label, remove_pad_tokens

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
            monitor=cfg["train"]["best_model_monitor"],
            mode="min" if cfg["train"]["best_model_monitor"] == "val_loss" else "max",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=cfg["train"]["earlystopping_monitor"],
            mode="min"
            if cfg["train"]["earlystopping_monitor"] == "val_loss"
            else "max",
            patience=cfg["train"]["patience"],
            verbose=True,
        ),
    ],
)

trainer.fit(model=model, datamodule=dataloader)
trainer.test(model=model, datamodule=dataloader, ckpt_path="best")

# validation data로 모델의 prediction 결과를 result 폴더에 csv파일로 저장합니다.
val_result = model.val_result
val_result["tokenized"] = remove_pad_tokens(
    val_result["tokenized"], tokenizer.pad_token
)
val_result["target"] = num_to_label(val_result["target"])
val_result["predict"] = num_to_label(val_result["predict"])
val_result_df = pd.DataFrame(val_result)
val_result_df.to_csv(
    cfg["dir"]["result_dir"] + result_name + "/val_result.csv", index=False
)
