import yaml
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn.functional as F

from models import BaseModel, ModelWithBinaryClassification, ModelWithEntityMarker
from loader import KLUEDataLoader
from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from utils import get_result_name, num_to_label, remove_pad_tokens,  tokenizer_update
from typing import Optional
import os
import wandb
import shutil, os
from pprint import pprint

# warning ignore(임시)
import warnings
warnings.filterwarnings(action='ignore')

def train(cfg, result_name :Optional[str] = None):
    # set random seed
    pl.seed_everything(cfg["seed"])

    # get result name
    if result_name is None:
        result_name = get_result_name()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], model_max_length=cfg["max_len"]
    )
    # marker, masking 옵션에 따른 tokenizer update
    tokenizer = tokenizer_update(tokenizer, cfg)

    model = eval(cfg['model_class'])(tokenizer, cfg)
    dataloader = KLUEDataLoader(tokenizer, cfg)

    # wandb logger, ggul_tiger 팀으로 run이 기록됩니다.
    logger = WandbLogger(
        name=result_name, 
        project=cfg['project_name'], 
        entity="ggul_tiger"
    )

    not_wanted_keys= ["num_workers", "train_dir", "test_dir", "result_dir", "val_size", "val_batch_size","test_batch_size","predict_batch_size", "min_epoch_to_log"]
    dict_filter = lambda item: False if item[0] in not_wanted_keys else True
    logged_cfg=dict(filter(dict_filter, cfg.items()))
    pprint(logged_cfg)
    logger.experiment.config.update(logged_cfg)
    print("WandbLogger id: {}".format(logger.version))

    trainer = pl.Trainer(
        accelerator="gpu",
        # strategy="ddp",
        max_epochs=cfg["epoch"],
        logger=logger,
        default_root_dir=cfg["result_dir"] + result_name,
        log_every_n_steps=50,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg["result_dir"] + result_name,
                filename="best_model",
                monitor=cfg["best_model_monitor"],
                mode="min" if cfg["best_model_monitor"] == "val_loss" else "max",
                save_top_k=1,
            ),
            EarlyStopping(
                monitor=cfg["earlystopping_monitor"],
                mode="min"
                if cfg["earlystopping_monitor"] == "val_loss"
                else "max",
                patience=cfg["patience"],
                verbose=True,
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            )
        ],
    )

    try:
        print('started!!!')
        trainer.fit(model=model, datamodule=dataloader)
    except KeyboardInterrupt as k:
        print("KeyboardInterrupt during fitting:", k)
    except TypeError as t:
        print("TypeError during fitting:", t)
    except Exception as e:
        print("Exception during fitting:", e)
    finally:
        print("current epoch:", trainer.current_epoch)
        if trainer.current_epoch > cfg["min_epoch_to_log"]:
            # test stage
            trainer.test(model=model, datamodule=dataloader, ckpt_path="best")
            # validation data로 모델의 prediction 결과를 result 폴더에 csv파일로 저장합니다.
            test_result = model.test_result
            test_result["tokenized"] = remove_pad_tokens(
                test_result["tokenized"], tokenizer.pad_token
            )
            test_result["target"] = num_to_label(test_result["target"])
            test_result["predict"] = num_to_label(test_result["predict"])
            test_result_df = pd.DataFrame(test_result)
            test_result_df.to_csv(
                cfg["result_dir"] + result_name + "/test_result.csv", index=False
            )

            # inference stage
            predictions = trainer.predict(
                model=model,
                datamodule=dataloader,
                ckpt_path="best",
            )  # list of prediction (batchs, batch_size, num_labels)

            # id_list = list(range(7765))
            probs_list = []
            label_list = []

            for output in predictions:
                probs_list.extend(output['probs'].tolist())
                label_list.extend(output['preds'].tolist())

            label_list = num_to_label(label_list)

            output = pd.DataFrame(
                {"id": list(range(len(label_list))), "pred_label": label_list, "probs": probs_list}
            )

            output.to_csv(
                cfg["result_dir"] + result_name + "/submission.csv", index=False
            )

            # dump config
            with open(cfg["result_dir"] + result_name + "/config.yaml", "w") as f:
                yaml.dump(cfg, f)

        else:
            wandb.finish()
            print('deleteing wandb run : {}'.format(logger.version))
            api = wandb.Api()
            run = api.run("ggul_tiger/KLUE/{}".format(logger.version))
            run.delete(delete_artifacts=True)

            if os.path.exists('results/{}'.format(result_name)):
                print('deleteing local folder : {}'.format(result_name))
                shutil.rmtree('results/{}'.format(result_name))

if __name__=="__main__":
    # load config
    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    train(cfg)