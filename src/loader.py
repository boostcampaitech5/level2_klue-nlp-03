import pytorch_lightning as pl
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from utils import label_to_num, load_data, preprocessing_dataset


class KLUEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.label = label_to_num(df["label"].to_list())
        self.tokenizer = tokenizer

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

    def tokenize(self, item: pd.Series) -> dict:
        """sub, obj entity, sentence를 이어붙이고 tokenize합니다."""
        joined_entity = "[SEP]".join([item["subject_entity"], item["object_entity"]])
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


class KLUEDataLoader(pl.LightningDataModule):
    def __init__(self, tokenzier, cfg: dict):
        super().__init__()
        self.tokenizer = tokenzier
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "fit":
            total_df = load_data(self.cfg["dir"]["train_dir"])
            total_df = preprocessing_dataset(total_df)
            train_df, val_df, _, _ = train_test_split(
                total_df,
                total_df["label"].values,
                test_size=self.cfg["train"]["val_size"],
                random_state=self.cfg["seed"],
            )
            self.train_dataset = KLUEDataset(train_df, self.tokenizer)
            self.val_dataset = KLUEDataset(val_df, self.tokenizer)

        if stage == "predict":
            predict_df = load_data(self.cfg["dir"]["test_dir"])
            predict_df = preprocessing_dataset(predict_df)
            self.predict_dataset = KLUEDataset(predict_df, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["train"]["batch_size"],
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
