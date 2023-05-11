import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoModel
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

        self.val_epoch_result = {
            "logits": torch.tensor([], dtype=torch.float32),
            "labels": torch.tensor([], dtype=torch.int64),
        }
        self.test_result = {
            "sentence": [],
            "tokenized": [],
            "target": [],
            "predict": [],
        }

    def configure_optimizers(self):
        MyOptim = eval("torch.optim." + self.cfg["optim"])
        optimizer = MyOptim(self.parameters(), lr=float(self.cfg["lr"]))
        if self.cfg["lr_scheduler"] is None:
            return [optimizer]
        else:
            scheduler = eval("torch.optim.lr_scheduler." + self.cfg["lr_scheduler"])
            return [optimizer], [scheduler]

    def compute_metrics(self, result):
        """loss와 score를 계산하는 함수"""
        probs = result["logits"]
        preds = torch.argmax(probs, dim=1)
        labels = result["labels"]
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

        loss = self.lossF(output["logits"], batch["labels"])

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )
        logits = output["logits"].detach().cpu()  # pt tensor (batch_size, num_labels)
        labels = batch["labels"].detach().cpu()  # pt tensor (batch_size)
        self.val_epoch_result["logits"] = torch.cat(
            (self.val_epoch_result["logits"], logits), dim=0
        )
        self.val_epoch_result["labels"] = torch.cat(
            (self.val_epoch_result["labels"], labels), dim=0
        )

    def on_validation_epoch_end(self):
        metrics = self.compute_metrics(self.val_epoch_result)
        self.log("val_loss", metrics["loss"], sync_dist=True)
        self.log("val_micro_F1_score", metrics["micro_F1_score"], sync_dist=True)
        self.log("val_auprc", metrics["auprc"], sync_dist=True)
        self.log("val_accuracy", metrics["accuracy"], sync_dist=True)
        self.val_epoch_result["logits"] = torch.tensor([], dtype=torch.float32)
        self.val_epoch_result["labels"] = torch.tensor([], dtype=torch.int64)

    def test_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        # 원래 문장, 원래 target, 모델의 prediction을 저장
        self.test_result["sentence"].extend(batch["sentence"])
        self.test_result["tokenized"].extend(
            self.tokenizer.batch_decode(batch["input_ids"].squeeze())
        )
        self.test_result["target"].extend(batch["labels"].tolist())
        self.test_result["predict"].extend(
            torch.argmax(output["logits"], dim=1).tolist()
        )

# test
class BinaryClassifier(BaseModel):
    ''' BinaryClassifier
    which picks up 'no-relation' or not
    '''
    def __init__(self, tokenizer, cfg: dict):
        super().__init__(tokenizer, cfg)
        self.model = AutoModel.from_pretrained(cfg["model_name"])
        self.lossBCE = torch.nn.BCEWithLogitsLoss()
        self.lossCE = eval("torch.nn." + cfg["loss"])()
        self.hidden_size = self.model.config.hidden_size
        self.pooler = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.multi_classifier = torch.nn.Linear(self.hidden_size, 30)
        self.bi_classifier = torch.nn.Linear(self.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.activation = torch.nn.Tanh()

        self.val_result = {
            # "sentence": [],
            "tokenized": [],
            "target": [],
            "predict": [],
        }

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        multiclf_token = outputs['pooler_output'] # (N, hdim)
        multiclf_token = self.dropout(multiclf_token)
        multiclf_token = self.multi_classifier(multiclf_token) # (N, 30)

        biclf_token = outputs['last_hidden_state'][:,1]
        biclf_token = self.pooler(biclf_token)
        biclf_token = self.activation(biclf_token)
        biclf_token = self.dropout(biclf_token)
        biclf_token = self.bi_classifier(biclf_token) # (N, 1)

        return {'multi':multiclf_token, 'bi':biclf_token}
        

    def compute_metrics(self, probs, labels):
        """loss와 score를 계산하는 함수"""
        probs = probs.detach().cpu()
        # preds = torch.argmax(probs, dim=1)
        preds = torch.where(probs >= 0.5, 1., 0.)
        labels = labels.detach().cpu()
        loss = self.lossCE(probs, labels)
        # calculate accuracy using sklearn's function
        # f1 = klue_re_micro_f1(preds, labels)
        # auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {
            "loss": loss,
            #  "micro_F1_score": None, "auprc": None,
                 "accuracy": acc}
    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )
        binary_labels = (batch['labels']!=0).bool().float().unsqueeze(1)
        loss = self.lossF(output['bi'], binary_labels)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        binary_labels = (batch['labels']!=0).bool().float().unsqueeze(1)
        metrics = self.compute_metrics(output['bi'], binary_labels)

        self.log("val_loss", metrics["loss"], sync_dist=True)
        # self.log("val_micro_F1_score", metrics["micro_F1_score"], sync_dist=True)
        # self.log("val_auprc", metrics["auprc"], sync_dist=True)
        self.log("val_accuracy", metrics["accuracy"], sync_dist=True)

    def test_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        binary_labels = (batch['labels']!=0).bool().float()

        # 원래 문장, 원래 target, 모델의 prediction을 저장
        # self.val_result["sentence"].extend(batch["sentence"])
        self.val_result["tokenized"].extend(
            self.tokenizer.batch_decode(batch["input_ids"].squeeze())
        )
        self.val_result["target"].extend(binary_labels.squeeze().tolist())
        self.val_result["predict"].extend(
           output['bi'].tolist()
        )