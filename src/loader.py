import pytorch_lightning as pl
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils import label_to_num, load_data, preprocessing_dataset
import re 


class KLUEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, input_format, model_class, save_sentence=False):
        self.df = df
        self.label = label_to_num(df["label"].to_list())
        self.tokenizer = tokenizer
        self.model_class = model_class
        self.input_format = input_format
        self.save_sentence = save_sentence

        if self.input_format not in ('default', 'entity_mask','entity_marker','entity_marker_punct','typed_entity_marker', 'typed_entity_marker_punct'):
            raise Exception("Invalid input format!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        tokenized_sentence = self.tokenize(item)
        ret_dict = {
            "input_ids": tokenized_sentence["input_ids"],
            "token_type_ids": tokenized_sentence["token_type_ids"],
            "attention_mask": tokenized_sentence["attention_mask"],
            "labels": torch.tensor(self.label[idx]),
        }
        if self.save_sentence:
            ret_dict["sentence"] = item["sentence"]
        return ret_dict
    
    # 임시
    # def get_tokenizer(self): return self.tokenizer
    # def len_tokenizer(self): return len(self.tokenizer)
    # def len_label(self): return len(self.label), len(list(set(self.label)))

    def tokenize(self, item: pd.Series) -> dict:
        """input format에 맞게 sub, obj entity, sentence를 이어붙이고 tokenize합니다."""
        # Case 00 : default (no masking or marking)
        if self.input_format == 'default':
            joined_entity = "[SEP]".join([item["subject_entity"]["word"], item["object_entity"]["word"]])
            # entity를 인식시켜주는 부분과 문장 부분을 서로 다른 token type id로 구별하기 위해서 joined_entity와 sentence를 따로 넣어줌
            tokenized_sentence = self.tokenizer(
                joined_entity,
                item["sentence"],
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokenized_sentence
        
        # Case 01 ~ 05
        subj_type = item["subject_entity"]["type"]
        obj_type = item["object_entity"]["type"]

        # tokienize item with masking or marking
        sent = item['sentence']

        subj_word = item["subject_entity"]["word"]
        obj_word = item["object_entity"]["word"]

        # preprocessing
        sent = self.preprocessing(sent)
        subj_word = self.preprocessing(subj_word)
        obj_word = self.preprocessing(obj_word)

        # Case 01 : entity_mask
        if self.input_format == 'entity_mask':
            subj_type = f'[S-{subj_type}]'
            obj_type = f'[O-{obj_type}]'
            sent = sent.replace(subj_word, subj_type)
            sent = sent.replace(obj_word, obj_type)

         # Case 02 : entity_marker
        elif self.input_format == 'entity_marker':
            subj_idx = sent.find(subj_word)
            sent = sent[:subj_idx] + '[E1]' + subj_word + '[/E1]' + sent[subj_idx+len(subj_word):]
            obj_idx = sent.find(obj_word)
            sent = sent[:obj_idx] + '[E2]' + obj_word + '[/E2]' + sent[obj_idx+len(obj_word):]

        # Case 03 : entity_marker_punct
        elif self.input_format == 'entity_marker_punct':
            subj_idx = sent.find(subj_word)
            sent = sent[:subj_idx] + '@' + subj_word + '@' + sent[subj_idx+len(subj_word):]
            obj_idx = sent.find(obj_word)
            sent = sent[:obj_idx] + '#' + obj_word + '#' + sent[obj_idx+len(obj_word):]

        # Case 04 : typed_entity_marker
        elif self.input_format == 'typed_entity_marker':
            # change format of subj/obj type 
            subj_type1 = '[S-{}]'.format(subj_type)
            subj_type2 = '[/S-{}]'.format(subj_type)
            obj_type1 = '[O-{}]'.format(obj_type)
            obj_type2 = '[/O-{}]'.format(obj_type)

            # add marker token
            subj_idx = sent.find(subj_word)
            sent = sent[:subj_idx] + subj_type1 + subj_word + subj_type2 + sent[subj_idx+len(subj_word):]
            obj_idx = sent.find(obj_word)
            sent = sent[:obj_idx] + obj_type1 + obj_word + obj_type2 + sent[obj_idx+len(obj_word):]

        # Case 05 : typed_entity_marker_punct
        elif self.input_format == 'typed_entity_marker_punct':
            # change format of subj/obj type 
            subj_type = subj_type.replace("_", " ").lower()
            obj_type = obj_type.replace("_", " ").lower()
            # add marker token
            subj_idx = sent.find(subj_word)
            sent = sent[:subj_idx] + '@ * ' + subj_type + ' * ' + subj_word + ' @' + sent[subj_idx+len(subj_word):]
            obj_idx = sent.find(obj_word)
            sent = sent[:obj_idx] + '# ^ ' + obj_type + ' ^ ' + obj_word + ' #' + sent[obj_idx+len(obj_word):]

           

        tokenized_sentence = self.tokenizer(
            sent,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return tokenized_sentence
    
    def preprocessing(self, sent):
        patterns = [
                (r'[#@^*]', ''),
                (r'\s+', ' ') # 이중 공백 제거
            ]

        for old, new in patterns:
            sent = re.sub(old, new, sent)
        return sent.strip()


class KLUEDataLoader(pl.LightningDataModule):
    def __init__(self, tokenizer, cfg: dict):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        
    # 임시
    def get_tokenizer(self): return self.tokenizer
    def len_tokenizer(self): return len(self.tokenizer)

    def setup(self, stage: str):
        if stage == "fit":
            # total_df = load_data(self.cfg["train_dir"])
            # total_df = preprocessing_dataset(total_df)
            # train_df, val_df = train_test_split(
            #     total_df,
            #     stratify=total_df["label"].values,
            #     test_size=self.cfg["val_size"],
            #     random_state=self.cfg["seed"],
            # )
            train_df = load_data(self.cfg["train_dir"])
            preprocessing_dataset(train_df)
            val_df = load_data(self.cfg["val_dir"])
            preprocessing_dataset(val_df)
            self.train_dataset = KLUEDataset(train_df, self.tokenizer, self.cfg['input_format'],self.cfg['model_class'])
            self.val_dataset = KLUEDataset(val_df, self.tokenizer, self.cfg['input_format'],self.cfg['model_class'], save_sentence=True)

        if stage == "predict":
            predict_df = load_data(self.cfg["test_dir"])
            predict_df = preprocessing_dataset(predict_df)
            self.predict_dataset = KLUEDataset(predict_df, self.tokenizer, self.cfg['input_format'], self.cfg['model_class'], save_sentence=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=True,
            batch_size=self.cfg["val_batch_size"],
            num_workers=self.cfg["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg["test_batch_size"],
            num_workers=self.cfg["num_workers"],
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.cfg["predict_batch_size"],
            num_workers=self.cfg["num_workers"],
        )
