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
            cfg["model_name"], num_labels=30
        )
        self.lossF = eval("torch.nn." + cfg["loss"])()
        self.val_result = {
            "sentence": [],
            "tokenized": [],
            "target": [],
            "predict": [],
        }

    def configure_optimizers(self):
        MyOptim = eval("torch.optim." + self.cfg["optim"])
        optimizer = MyOptim(self.parameters(), lr=float(self.cfg["lr"]))
        if self.cfg['lr_scheduler'] is None:
            return [optimizer]
        else:
            scheduler = eval('torch.optim.lr_scheduler.'+self.cgf['lr_scheduler'])
            return [optimizer], [scheduler]

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
        )

        loss = self.lossF(output.logits, batch["labels"])

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        metrics = self.compute_metrics(output, batch["labels"])

        self.log("val_loss", metrics["loss"], sync_dist=True)
        self.log("val_micro_F1_score", metrics["micro_F1_score"], sync_dist=True)
        self.log("val_auprc", metrics["auprc"], sync_dist=True)
        self.log("val_accuracy", metrics["accuracy"], sync_dist=True)

    def test_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        # 원래 문장, 원래 target, 모델의 prediction을 저장
        self.val_result["sentence"].extend(batch["sentence"])
        self.val_result["tokenized"].extend(
            self.tokenizer.batch_decode(batch["input_ids"].squeeze())
        )
        self.val_result["target"].extend(batch["labels"].tolist())
        self.val_result["predict"].extend(
            torch.argmax(output["logits"], dim=1).tolist()
        )

# test
class BertForDuoClassifier(BaseModel):
    ''' BinaryClassifier -> MultiClassifier
    https://www.kaggle.com/code/duongthanhhung/bert-relation-extraction
    '''
    def __init__(self, tokenizer, cfg: dict):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=1
        )

    def compute_metrics(self, output, labels):
        """loss와 score를 계산하는 함수"""
        probs = output.logits.detach().cpu()
        preds = torch.argmax(probs, dim=1)
        labels = labels.detach().cpu()
        loss = self.lossF(probs, labels)
        # calculate accuracy using sklearn's function
        # f1 = klue_re_micro_f1(preds, labels)
        # auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {
            "loss": loss,
            #  "micro_F1_score": f1, "auprc": auprc,
                 "accuracy": acc}