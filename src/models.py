from typing import Any
import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
from utils import klue_re_auprc, klue_re_micro_f1
from sklearn.metrics import accuracy_score


class BaseModel(pl.LightningModule):
    def __init__(self, tokenizer, cfg: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["train"]["model_name"], num_labels=30
        )
        self.lossF = eval("torch.nn." + cfg["train"]["loss"])()

    def configure_optimizers(self):
        MyOptim = eval("torch.optim." + self.cfg["train"]["optim"])
        optimizer = MyOptim(self.parameters(), lr=float(self.cfg["train"]["lr"]))

        return [optimizer]

    def compute_metrics(self, output, labels):
        """loss와 score를 계산하는 함수"""
        probs = output.logits.detach().cpu()
        preds = torch.argmax(probs, dim=1)
        labels = labels.detach().cpu()
        loss = self.lossF(probs, labels)
        # calculate accuracy using sklearn's function
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {"loss": loss, "micro_F1_score": f1, "auprc": auprc, "accuracy": acc}

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
            labels=batch["labels"],
        )

        loss = self.lossF(output.logits, batch["labels"])

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
            labels=batch["labels"],
        )

        metrics = self.compute_metrics(output, batch["labels"])

        self.log("val_loss", metrics["loss"])
        self.log("val_micro_F1_score", metrics["micro_F1_score"])
        self.log("val_auprc", metrics["auprc"])
        self.log("val_accuracy", metrics["accuracy"])
