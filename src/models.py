import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoModel
from utils import klue_re_auprc, klue_re_micro_f1, model_freeze
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class BaseModel(pl.LightningModule):
    def __init__(self, tokenizer, cfg: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=30
        )
        self.model_resize()
        self.lossF = eval("torch.nn." + cfg["loss"])(label_smoothing=self.cfg['label_smoothing'])

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
    
    def model_resize(self):
        before = self.model.config.vocab_size
        self.model.resize_token_embeddings(len(self.tokenizer))
        after = self.model.config.vocab_size
        print(f"Model input format : {self.cfg['input_format']}")
        if before != after:
            print(f'Model vocab_size changed : {before} -> {after}')
        else:
            print(f"Model vocab size : {after}")

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
        
    def lr_lambda(self, current_epoch):
        """custom lr_scheduler: linear하게 상승 후 하강"""
        growth_ratio = 0.3  # 증가하는 구간 (30%), 하강하는 구간(70%)
        epsilon = 0.05
        max_epochs = self.cfg['epoch']
        if current_epoch <= int(max_epochs * growth_ratio):
            # 증가 구간
            return current_epoch / (max_epochs * growth_ratio) + epsilon
        else:
            # 감소 구간
            return (max_epochs - current_epoch) / (max_epochs * (1 - growth_ratio)) + epsilon

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
    
    def predict_step(self, batch, batch_idx):
        output = self.forward(batch)
        probs = F.softmax(output['logits'],dim=1)
        preds = torch.argmax(probs, dim=1)
        return {'preds':preds, 'probs':probs}

class ModelWithEntityMarker(BaseModel):
    ''' ModelWithEntityMarker
    Classifier with CLS tokens, entity marker(@, #) tokens
    '''
    def __init__(self, tokenizer, cfg: dict, num_labels=30):
        super().__init__(tokenizer, cfg)
        self.pooling_type = cfg['pooling_type'] # "entity_start_token", "entity_start_end_token", "entity_tokens"
        self.input_format = cfg['input_format']
        self.model = AutoModel.from_pretrained(cfg["model_name"])
        self.model_resize()
        # self.model = model_freeze(self.model)
        self.lossF = eval("torch.nn." + cfg["loss"])(label_smoothing=self.cfg['label_smoothing'])

        self.hidden_size = self.model.config.hidden_size
        self.pooler = torch.nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.classifier = torch.nn.Linear(self.hidden_size, num_labels, bias=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.activation = torch.nn.Tanh()
        if  cfg['input_format'] != 'default':
            subj_markers = '@'
            obj_markers = '#'
            # subj_markers = '@ [E1] [/E1] [S-PER] [S-ORG] [/S-PER] [/S-ORG]' 
            # obj_markers = '# [E2] [/E2] [O-PER] [O-ORG] [O-LOC] [O-NOH] [O-POH] [O-DAT] [/O-PER] [/O-ORG] [/O-LOC] [/O-NOH] [/O-POH] [/O-DAT]'
            self.subj_ids = {k:v for k,v in zip(subj_markers.split(),self.tokenizer(subj_markers, add_special_tokens=False)['input_ids'])}
            self.obj_ids = {k:v for k,v in zip(obj_markers.split(),self.tokenizer(obj_markers, add_special_tokens=False)['input_ids'])}
        else:
            self.subj_ids = self.obj_ids = {tokenizer.sep_token:tokenizer.sep_token_id}

    def forward(self, input):
        outputs = self.model(
            input_ids=input["input_ids"].squeeze(),
            token_type_ids=input["token_type_ids"].squeeze(),
            attention_mask=input["attention_mask"].squeeze(),
        )
        outputs = self.pooling(input['input_ids'], outputs['last_hidden_state'])
        outputs = self.pooler(outputs)
        outputs = self.activation(outputs)

        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
             
        return {'logits':outputs}
    
    def pooling(self, batch_input_ids, last_hidden_state):
        pooler_output = torch.Tensor().to(self.device)
        if self.pooling_type == 'none':
            return torch.cat([last_hidden_state[:,0],last_hidden_state[:,0]], dim=1)

        for i, input_ids in enumerate(batch_input_ids):
            
            subj_idx, obj_idx = self.get_marker_index(input_ids.squeeze())
            try:
                if self.pooling_type == 'entity_start_token': # (1, hdim * 2)
                    hidden_states = torch.cat([
                        last_hidden_state[i, subj_idx[0]].view(-1, self.hidden_size),
                        last_hidden_state[i, obj_idx[0]].view(-1, self.hidden_size)
                    ], dim=1) #(1, hdim * 2)
                elif self.pooling_type == 'entity_start_end_token': # (1, hdim * 2)
                    hidden_states = torch.cat([
                        (last_hidden_state[i, subj_idx[0]] + last_hidden_state[i, subj_idx[1]]).view(-1, self.hidden_size) / 2, 
                        (last_hidden_state[i, obj_idx[0]] + last_hidden_state[i, obj_idx[1]]).view(-1, self.hidden_size) / 2, 
                    ], dim=1)  
                elif self.pooling_type == 'entity_tokens': # (1, hdim * 2)
                    hidden_states = torch.cat([
                        torch.mean(last_hidden_state[i, subj_idx[0]:subj_idx[1] + 1].view(-1, self.hidden_size),dim=0).unsqueeze(0),
                        torch.mean(last_hidden_state[i, obj_idx[0]:obj_idx[1] + 1].view(-1, self.hidden_size),dim=0).unsqueeze(0)
                    ], dim=1)
                else:
                    hidden_states = last_hidden_state[i, 0].unsqueeze(0)
            except Exception as e:
                print('Error:', e)
                hidden_states = torch.cat([
                last_hidden_state[i,0].unsqueeze(0),
                last_hidden_state[i,(subj_idx+obj_idx)[0]].unsqueeze(0)
                ], dim=1) 
                    
            pooler_output = torch.cat([pooler_output, hidden_states],dim=0)

        return pooler_output

    def get_marker_index(self, input_ids):
        """ entity_marker, entity_mask"""
        subj_idx = []
        obj_idx = []
        n = 2 if self.input_format not in ['default', 'entity_mask'] else 1
        for i, ids in enumerate(input_ids):
            if ids in self.subj_ids.values():
                subj_idx.append(i)
            elif ids in self.obj_ids.values():
                obj_idx.append(i)
            if len(subj_idx)==n and len(obj_idx)==n:
                break
        idx = subj_idx + obj_idx
        if len(subj_idx)==n and len(obj_idx)==n: return (subj_idx, obj_idx)
        elif len(subj_idx)==n or len(obj_idx)==n: return  (idx[:2], idx[:2])
        else: return ([0,0], [0,0])

    
class BinaryClassifier(ModelWithEntityMarker):
    ''' BinaryClassifier
    which picks up 'no-relation' or not
    '''
    def __init__(self, tokenizer, cfg: dict, num_labels=1):
        super().__init__(tokenizer, cfg, num_labels=1)
        self.lossF = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.classifier = torch.nn.Linear(self.hidden_size, num_labels)

    def compute_metrics(self, result):
        """loss와 score를 계산하는 함수"""
        logits = result["logits"]
        preds = torch.where(logits>=0.5, 1., 0.)
        labels = result["labels"]
        loss = self.lossF(logits, labels)
        # calculate accuracy using sklearn's function
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {"loss": loss, "accuracy": acc}

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        logits = self.sigmoid(output['logits'])
        labels = (batch['labels']!=0).bool().float().unsqueeze(1)
        loss = self.lossF(logits, labels)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        logits = self.sigmoid(output["logits"]).detach().cpu() 
        labels = (batch['labels']!=0).bool().float().unsqueeze(1)
        labels = labels.detach().cpu() 
        self.val_epoch_result["logits"] = torch.cat(
            (self.val_epoch_result["logits"], logits), dim=0
        )
        self.val_epoch_result["labels"] = torch.cat(
            (self.val_epoch_result["labels"], labels), dim=0
        )

    def on_validation_epoch_end(self):
        metrics = self.compute_metrics(self.val_epoch_result)
        self.log("val_loss", metrics["loss"], sync_dist=True)
        self.log("val_accuracy", metrics["accuracy"], sync_dist=True)
        self.val_epoch_result["logits"] = torch.tensor([], dtype=torch.float32)
        self.val_epoch_result["labels"] = torch.tensor([], dtype=torch.int64)

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        logits = self.sigmoid(output['logits'])
        labels = (batch['labels']!=0).bool().float().squeeze()

        # 원래 문장, 원래 target, 모델의 prediction을 저장
        self.test_result["sentence"].extend(batch["sentence"])
        self.test_result["tokenized"].extend(
            self.tokenizer.batch_decode(batch["input_ids"].squeeze())
        )
        self.test_result["target"].extend(labels.tolist())
        self.test_result["predict"].extend(
            torch.where(logits>0.5, 1, 0).tolist()
        )
    
    def predict_step(self, batch, batch_idx):
        output = self.forward(batch)
        logits = self.sigmoid(output['logits'])
        other_probs = ((1-output['logits'])/29).expand(-1,29)
        probs = torch.cat([logits, other_probs], dim=1)
        assert probs.size(-1) == 30, 'lael size should be 30'
        preds = torch.where(logits>=0.5, 1., 0.).squeeze()
        return {'preds':preds, 'probs':probs}

class TripleClassifier(ModelWithEntityMarker):
    ''' TripleClassifier
    no-relation or per or org
    '''
    def __init__(self, tokenizer, cfg: dict, num_labels=3):
        super().__init__(tokenizer, cfg, num_labels=3)
        self.lossF = eval("torch.nn." + cfg["loss"])(
            weight=torch.Tensor([cfg['class_weight'],1.0, 1.0]),
            label_smoothing=self.cfg['label_smoothing']
            )
        self.classifier = torch.nn.Linear(self.hidden_size, num_labels)
        self.test_result = {
            "sentence": [],
            "tokenized": [],
            "target": [],
            "predict": [],
            "probs":[]
        }

    def compute_metrics(self, result):
        """loss와 score를 계산하는 함수"""
        logits = result["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        labels = result["labels"]
        loss = self.lossF(logits.to(self.device), labels.to(self.device))
        # calculate accuracy using sklearn's function
        precision, recall, f1, _ =  precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {"loss": loss.detach().cpu(), "micro_F1_score": f1, "precision": precision, "accuracy": acc}
    def on_validation_epoch_end(self):
        metrics = self.compute_metrics(self.val_epoch_result)
        self.log("val_loss", metrics["loss"], sync_dist=True)
        self.log("val_micro_F1_score", metrics["micro_F1_score"], sync_dist=True)
        self.log("val_precision", metrics["precision"], sync_dist=True)
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
        self.test_result["probs"].extend(
            F.softmax(output["logits"], dim=1).tolist()
        )