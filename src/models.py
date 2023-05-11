import torch
import torch.nn.functional as F
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

    def forward(self, input):
        return self.model(
            input_ids=input["input_ids"].squeeze(),
            token_type_ids=input["token_type_ids"].squeeze(),
            attention_mask=input["attention_mask"].squeeze(),
        )

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
        logits = result["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        labels = result["labels"]
        loss = self.lossF(logits, labels)
        # calculate accuracy using sklearn's function
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {"loss": loss, "micro_F1_score": f1, "auprc": auprc, "accuracy": acc}

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.lossF(output["logits"], batch["labels"])

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
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
        output = self.forward(batch)

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
class ModelWithBinaryClassification(BaseModel):
    ''' ModelWithBinaryClassification
    which picks up 'no-relation' or not
    '''
    def __init__(self, tokenizer, cfg: dict):
        super().__init__(tokenizer, cfg)
        self.model = AutoModel.from_pretrained(cfg["model_name"])
        self.lossBCE = torch.nn.BCEWithLogitsLoss()
        self.lossCE = eval("torch.nn." + cfg["loss"])()
        self.hidden_size = self.model.config.hidden_size
        self.pooler = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.multi_classifier = torch.nn.Linear(self.hidden_size*2, 30)
        self.bi_classifier = torch.nn.Linear(self.hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.activation = torch.nn.Tanh()
        self.alpha = 0.6


    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        multiclf_token = self.dropout(outputs['pooler_output']) # (N, hdim)

        biclf_token = outputs['last_hidden_state'][:,1]
        biclf_token = self.pooler(biclf_token)
        biclf_token = self.activation(biclf_token) # (N, hdim)
        biclf = self.dropout(biclf_token)
        biclf = self.bi_classifier(biclf) # (N, 1)

        multiclf = torch.cat([multiclf_token,biclf_token],dim=1) # (N,hdim*2)
        multiclf = self.multi_classifier(multiclf) # (N, 30)

        return {'multi':multiclf, 'bi':biclf}
        

    def compute_metrics(self, result):
        """loss와 score를 계산하는 함수"""
        probs = result["logits"]
        preds = torch.argmax(probs, dim=1)
        labels = result["labels"]
        # loss = self.lossF(probs, labels)
        # calculate accuracy using sklearn's function
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {
            # "loss": loss,
             "micro_F1_score": f1, "auprc": auprc,
                 "accuracy": acc}
    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )

        binary_labels = (batch['labels']!=0).bool().float().unsqueeze(1)

        loss_bi = self.lossBCE(output['bi'], binary_labels) * self.alpha
        loss_mul = self.lossCE(output['multi'], batch['labels']) * (1-self.alpha)
        loss = loss_bi + loss_mul

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"].squeeze(),
            token_type_ids=batch["token_type_ids"].squeeze(),
            attention_mask=batch["attention_mask"].squeeze(),
        )
        binary_labels = (batch['labels']!=0).bool().float().unsqueeze(1)

        loss_bi = self.lossBCE(output['bi'], binary_labels) * self.alpha
        loss_mul = self.lossCE(output['multi'], batch['labels']) * (1-self.alpha)
        loss = loss_bi + loss_mul

        logits = output['multi'].detach().cpu()
        labels = batch['labels'].detach().cpu()

        self.log("val_loss", loss, sync_dist=True, on_epoch=True)
        self.val_epoch_result["logits"] = torch.cat(
            (self.val_epoch_result["logits"], logits), dim=0
        )
        self.val_epoch_result["labels"] = torch.cat(
            (self.val_epoch_result["labels"], labels), dim=0
        )

    def on_validation_epoch_end(self):
        metrics = self.compute_metrics(self.val_epoch_result)
        # self.log("val_loss", metrics["loss"], sync_dist=True)
        self.log("val_micro_F1_score", metrics["micro_F1_score"], sync_dist=True)
        self.log("val_auprc", metrics["auprc"], sync_dist=True)
        self.log("val_accuracy", metrics["accuracy"], sync_dist=True)
        self.val_epoch_result["logits"] = torch.tensor([], dtype=torch.float32)
        self.val_epoch_result["labels"] = torch.tensor([], dtype=torch.int64)

    def test_step(self, batch, batch_idx):
        output = self(
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
            torch.argmax(output['multi'], dim=1).tolist()
        )
