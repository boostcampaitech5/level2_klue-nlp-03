import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import custom_loss
from transformers import AutoModelForSequenceClassification, AutoModel
from utils import klue_re_auprc, klue_re_micro_f1, model_freeze, label_to_num
from sklearn.metrics import accuracy_score


class BaseModel(pl.LightningModule):
    def __init__(self, tokenizer, cfg: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=30
        )
        self.model_resize()
        self.lossF = eval(cfg["loss"])
        class_weight = torch.tensor(cfg["class_weight"], dtype=torch.float32).to("cuda:0")

        if cfg["loss"] == "nn.CrossEntropyLoss":
            self.lossF = self.lossF(weight=class_weight, label_smoothing=self.cfg["label_smoothing"])
        elif cfg["loss"] == "custom_loss.FocalLoss":
            self.lossF = self.lossF(alpha=class_weight, gamma=cfg["FocalLoss_gamma"])

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
        labels = result["labels"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # calculate accuracy using sklearn's function
        f1 = klue_re_micro_f1(preds, labels)
        auprc = klue_re_auprc(probs, labels)
        acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

        return {"micro_F1_score": f1, "auprc": auprc, "accuracy": acc}

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.lossF(output["logits"], batch["labels"])

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.lossF(output["logits"], batch["labels"])
        logits = output["logits"].detach().cpu()  # pt tensor (batch_size, num_labels)
        labels = batch["labels"].detach().cpu()  # pt tensor (batch_size)
        self.log("val_loss", loss, on_epoch=True)
        self.val_epoch_result["logits"] = torch.cat(
            (self.val_epoch_result["logits"], logits), dim=0
        )
        self.val_epoch_result["labels"] = torch.cat(
            (self.val_epoch_result["labels"], labels), dim=0
        )

    def on_validation_epoch_end(self):
        metrics = self.compute_metrics(self.val_epoch_result)
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
                elif self.pooing_type == 'entity_start_end_token': # (1, hdim * 2)
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
    

class RECENT(ModelWithEntityMarker):
    def __init__(self, tokenizer, cfg: dict, num_labels=30):
        super().__init__(tokenizer, cfg, num_labels)

        self.restrict_dict = {
            "ORG_PER": ["no_relation", "org:founded_by", "org:top_members/employees"],
            "ORG_ORG": ["no_relation", "org:founded_by", "org:member_of", "org:political/religious_affiliation", "org:members", "org:alternate_names", "org:place_of_headquarters", "org:product"],
            "ORG_DAT": ["no_relation", "org:dissolved", "org:founded"], 
            "ORG_LOC": ["no_relation", "org:member_of", "org:place_of_headquarters", "org:members", "org:political/religious_affiliation", "org:product"],
            "ORG_POH": ["no_relation", "org:alternate_names", "org:product", "org:political/religious_affiliation", "org:member_of", "org:members"],
            "ORG_NOH": ["no_relation", "org:number_of_employees/members"],
            "PER_PER": ["no_relation", "per:colleagues", "per:spouse", "per:children", "per:parents", "per:other_family", "per:siblings", "per:alternate_names", "per:employee_of", "per:product"],
            "PER_ORG": ["no_relation", "per:employee_of", "per:schools_attended", "per:origin", "per:religion", "per:title", "per:place_of_death", "per:product", "per:place_of_residence"],
            "PER_DAT": ["no_relation", "per:date_of_birth", "per:date_of_death", "per:origin"],
            "PER_LOC": ["no_relation", "per:place_of_birth", "per:place_of_residence", "per:place_of_death", "per:origin", "per:employee_of", "per:title"],
            "PER_POH": ["no_relation", "per:title", "per:alternate_names", "per:product", "per:employee_of", "per:origin", "per:parents", "per:place_of_death", "per:religion", "per:siblings", "per:children", "per:place_of_residence"],
            "PER_NOH": ["no_relation"],
            } # subject, object type에 따라서 나올 수 있는 label을 제한하는 dictionary
        
        self.type_pair_to_num_label = dict() # restrict dict의 label을 숫자로 바꾼 dictionary

        for key, value in zip(self.restrict_dict.keys(), self.restrict_dict.values()):
            self.type_pair_to_num_label[key] = label_to_num(value)

        self.classifier_list = torch.nn.ModuleList()
        for type_pair, labels in zip(self.restrict_dict.keys(), self.restrict_dict.values()):
            classifier = torch.nn.Sequential(torch.nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=True), 
                                             torch.nn.Tanh(),
                                             torch.nn.Dropout(p=0.1, inplace=False),
                                             torch.nn.Linear(in_features=self.hidden_size, out_features=len(labels), bias=True))
            self.classifier_list.add_module(name=type_pair, module=classifier)
        
    def forward(self, input):
        outputs = self.model(
            input_ids=input["input_ids"].squeeze(),
            token_type_ids=input["token_type_ids"].squeeze(),
            attention_mask=input["attention_mask"].squeeze(),
        )

        pooled_hidden_state = self.pooling(input['input_ids'], outputs['last_hidden_state']) # dim = (batch_size, 2 * hiddendim)

        logits_list = []

        for i in range(pooled_hidden_state.shape[0]):
            type_pair = f"{input['subject_type'][i]}_{input['object_type'][i]}"

            classifier_output = self.classifier_list.get_submodule(type_pair)(pooled_hidden_state[i]) # output dimension depends on type_pair. dim = (number of possible classes,)

            logits = torch.empty(30).fill_(classifier_output.min() - classifier_output.abs().mean()).to(self.device) # fills the logit value of a not related class about type_pair with a low value.
            logits[self.type_pair_to_num_label[type_pair]] = classifier_output
            logits_list.append(logits)

        return {"logits": torch.stack(logits_list)}


# test code
if __name__ == "__main__":
    import yaml
    from utils import load_data, preprocessing_dataset
    from loader import KLUEDataset
    from transformers import AutoTokenizer

    with open("./config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    model = RECENT(tokenizer, cfg)

    train_df = load_data(cfg["train_dir"])
    train_df = preprocessing_dataset(train_df)
    train_dataset = KLUEDataset(train_df, tokenizer, "entity_marker_punct", cfg['model_class'])

    print(model.forward(train_dataset[0]))