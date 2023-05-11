import pytorch_lightning as pl
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils import label_to_num, load_data, preprocessing_dataset


class KLUEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, input_format):
        self.df = df
        self.label = label_to_num(df["label"].to_list())
        self.tokenizer = tokenizer

        self.input_format = input_format
        # self.new_tokens = []
        # if self.input_format == 'entity_marker': # -> utils marker_tokenizer_update 대체
        #     self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        # self.tokenizer.add_tokens(self.new_tokens)
        if self.input_format not in ('default', 'entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct'):
            raise Exception("Invalid input format!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        tokenized_sentence = self.tokenize(item)
        ret_dict = {
            "sentence": item["sentence"],
            "input_ids": tokenized_sentence["input_ids"],
            "token_type_ids": tokenized_sentence["token_type_ids"],
            "attention_mask": tokenized_sentence["attention_mask"],
            "labels": torch.tensor(self.label[idx]),
        }

        return ret_dict
    
    # 임시
    def get_tokenizer(self): return self.tokenizer
    def len_tokenizer(self): return len(self.tokenizer)
    def len_label(self): return len(self.label), len(list(set(self.label)))

    def tokenize(self, item: pd.Series) -> dict:
        """sub, obj entity, sentence를 이어붙이고 tokenize합니다."""
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
        
        # case need to append new_token in tokenizer
        # Case 01, 04
        subj_type = item["subject_entity"]["type"]
        obj_type = item["object_entity"]["type"]

        # Case 04 : typed_entity_marker -> utils marker_tokenizer_update 대체
        if self.input_format == 'typed_entity_marker':
            subj_start = '[SUBJ-{}]'.format(subj_type)
            subj_end = '[/SUBJ-{}]'.format(subj_type)
            obj_start = '[OBJ-{}]'.format(obj_type)
            obj_end = '[/OBJ-{}]'.format(obj_type)

        # case need to change format of subj/obj type
        # Case 05 : typed_entity_marker_punct
        elif self.input_format == 'typed_entity_marker_punct':
            subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower()) 
            obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())


        # tokienize item with masking or marking
        # Case 01 ~ 05
        sent = item['sentence']
        subj_word = item["subject_entity"]["word"]
        obj_word = item["object_entity"]["word"]

        # Case 01 : entity_mask
        if self.input_format == 'entity_mask':
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
            sent = sent.replace(subj_word, subj_start + ' ' + subj_type + ' ' + subj_end)
            sent = sent.replace(obj_word, obj_start + ' ' + obj_type + ' ' + obj_end)

        # Case 05 : typed_entity_marker_punct
        elif self.input_format == 'typed_entity_marker_punct':
            subj_idx = sent.find(subj_word)
            sent = sent[:subj_idx] + '@ * ' + subj_word + '* @' + sent[subj_idx+len(subj_word):]
            obj_idx = sent.find(obj_word)
            sent = sent[:obj_idx] + '# ^ ' + obj_word + ' ^ #' + sent[obj_idx+len(obj_word):]
        
        tokenized_sentence = self.tokenizer(
            sent,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return tokenized_sentence


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
            total_df = load_data(self.cfg["train_dir"])
            total_df = preprocessing_dataset(total_df)
            train_df, val_df, _, _ = train_test_split(
                total_df,
                total_df["label"].values,
                test_size=self.cfg["val_size"],
                random_state=self.cfg["seed"],
            )
            self.train_dataset = KLUEDataset(train_df, self.tokenizer, self.cfg['input_format'])
            self.val_dataset = KLUEDataset(val_df, self.tokenizer, self.cfg['input_format'])

        if stage == "predict":
            predict_df = load_data(self.cfg["test_dir"])
            predict_df = preprocessing_dataset(predict_df)
            self.predict_dataset = KLUEDataset(predict_df, self.tokenizer, self.cfg['input_format'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            num_workers=self.cfg["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
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
