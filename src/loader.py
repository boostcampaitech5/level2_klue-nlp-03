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
            "subject_type": tokenized_sentence["subject_type"],
            "object_type": tokenized_sentence["object_type"],
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
            
        type_convert = {'PER':'사람', 'ORG':'조직','LOC':'지역','NOH':'숫자','POH':'기타','DAT':'날짜'}
        try: 
            subj_type = type_convert[item["subject_entity"]["type"]]
            obj_type = type_convert[item["object_entity"]["type"]]
        except: 
            subj_type = item["subject_entity"]["type"]
            obj_type = item["object_entity"]["type"]

        if subj_type in ["지역", "숫자", "기타", "날짜"]: # subject type을 사람과 조직으로 제한
            subj_type = "조직"

        # tokienize item with masking or marking
        sent = item['sentence']

        subj_word = item["subject_entity"]["word"]
        subj_start = item["subject_entity"]["start_idx"]
        subj_end = item["subject_entity"]["end_idx"]+1

        obj_word = item["object_entity"]["word"]
        obj_start = item["object_entity"]["start_idx"]
        obj_end = item["object_entity"]["end_idx"]+1

        # # 일단 word가 등장하는 모든 index를 구하고 몇번째 등장하는 word가 subject/object인지 subj_idx/obj_idx에 저장
        # subj_idx=None
        # for i, match in enumerate(re.finditer(subj_word, sent)):
        #     start,e = match.span()
        #     if start==subj_start:
        #         subj_idx = i
        #         break

        # obj_idx=None
        # for i, match in enumerate(re.finditer(obj_word, sent)):
        #     start,e = match.span()
        #     if start==obj_start:
        #         obj_idx = i
        #         break

        # preprocessing
        sent = self.preprocessing(sent)
        subj_word = self.preprocessing(subj_word)
        obj_word = self.preprocessing(obj_word)

        # subj_matches = list(re.finditer(re.escape(subj_word), sent))
        # subj_start, subj_end = subj_matches[subj_idx].span()
        # obj_matches = list(re.finditer(re.escape(obj_word), sent))
        # obj_start, obj_end = obj_matches[obj_idx].span()

        # Case 01 : entity_mask
        if self.input_format == 'entity_mask':
            subj_type_mask = f'[S-{subj_type}]'
            obj_type_mask = f'[O-{obj_type}]'
            if subj_start < obj_start:
                # subject가 먼저 등장, 뒤에 오는 object부터 처리
                sent = sent[:obj_start] + obj_type_mask + sent[obj_end:]
                sent = sent[:subj_start] + subj_type_mask + sent[subj_end:]
            else:
                # object가 먼저 등장, 뒤에 오는 subject부터 처리
                sent = sent[:subj_start] + subj_type_mask + sent[subj_end:]
                sent = sent[:obj_start] + obj_type_mask + sent[obj_end:]

         # Case 02 : entity_marker
        elif self.input_format == 'entity_marker':
            if subj_start < obj_start:
                # subject가 먼저 등장, 뒤에 오는 object부터 처리
                sent = sent[:obj_start]+'[E2]'+obj_word+'[/E2]'+sent[obj_end:]
                sent = sent[:subj_start]+'[E1]'+subj_word+'[/E1]'+sent[subj_end:]
            else:
                # object가 먼저 등장, 뒤에 오는 subject부터 처리
                sent = sent[:subj_start]+'[E1]'+subj_word+'[/E1]'+sent[subj_end:]
                sent = sent[:obj_start]+'[E2]'+obj_word+'[/E2]'+sent[obj_end:]

        # Case 03 : entity_marker_punct
        elif self.input_format == 'entity_marker_punct':
            if subj_start < obj_start:
                # subject가 먼저 등장, 뒤에 오는 object부터 처리
                sent = sent[:obj_start] + '#' + obj_word + '#' + sent[obj_end:]
                sent = sent[:subj_start] + '@' + subj_word + '@' + sent[subj_end:]
            else:
                sent = sent[:subj_start] + '@' + subj_word + '@' + sent[subj_end:]
                sent = sent[:obj_start] + '#' + obj_word + '#' + sent[obj_end:]


        # Case 04 : typed_entity_marker
        elif self.input_format == 'typed_entity_marker':
            # change format of subj/obj type 
            subj_type1 = '[S-{}]'.format(subj_type)
            subj_type2 = '[/S-{}]'.format(subj_type)
            obj_type1 = '[O-{}]'.format(obj_type)
            obj_type2 = '[/O-{}]'.format(obj_type)

            # add marker token
            if subj_start < obj_start:
                # subject가 먼저 등장, 뒤에 오는 object부터 처리
                sent = sent[:obj_start] + obj_type1 + obj_word + obj_type2 + sent[obj_end:]
                sent = sent[:subj_start] + subj_type1 + subj_word + subj_type2 + sent[subj_end:]
            else:
                sent = sent[:subj_start] + subj_type1 + subj_word + subj_type2 + sent[subj_end:]
                sent = sent[:obj_start] + obj_type1 + obj_word + obj_type2 + sent[obj_end:]

        # Case 05 : typed_entity_marker_punct
        elif self.input_format == 'typed_entity_marker_punct':
            # change format of subj/obj type 
            subj_type = subj_type.replace("_", " ").lower()
            obj_type = obj_type.replace("_", " ").lower()
            # add marker token
            if subj_start < obj_start:
                # subject가 먼저 등장, 뒤에 오는 object부터 처리
                sent = sent[:obj_start] + '# ^ ' + obj_type + ' ^ ' + obj_word + ' #' + sent[obj_end:]
                sent = sent[:subj_start] + '@ * ' + subj_type + ' * ' + subj_word + ' @' + sent[subj_end:]
            else:
                sent = sent[:subj_start] + '@ * ' + subj_type + ' * ' + subj_word + ' @' + sent[subj_end:]
                sent = sent[:obj_start] + '# ^ ' + obj_type + ' ^ ' + obj_word + ' #' + sent[obj_end:]
        
        # print(sent)

        tokenized_sentence = self.tokenizer(
            sent,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokenized_sentence["subject_type"] = subj_type
        tokenized_sentence["object_type"] = obj_type
        
        return tokenized_sentence
    
    def preprocessing(self, sent:str)->str:
        """구두점 및 이중공백 제거

        Args:
            sent (str): _description_

        Returns:
            str: _description_
        """
        patterns = [
                (r'[#@]', '-'),
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
            train_df = load_data(self.cfg["train_dir"])
            train_df = preprocessing_dataset(train_df)
            val_df = load_data(self.cfg["val_dir"])
            val_df = preprocessing_dataset(val_df)

            # RECENT 모델은 subject-object type pair에 따라서 독립적인 여러 개의 classifier가 있습니다.
            # 그렇기 때문에 한 batch 안에 여러 종류의 type pair가 들어가면, 어떤 classifier는 아주 적은 데이터로 loss를 계산하고 학습하게 됩니다.
            # (batch size가 작아지는 효과와 같습니다.)
            # 그렇기 때문에 type pair로 정렬해줍니다. 이렇게 되면 한 batch 안에 최대한 적은 종류의 type pair가 들어가게 됩니다.
            if self.cfg["model_class"] == "RECENT":
                # train_df["type_pair"] = train_df.apply(lambda x: f"{x['subject_entity']['type']}_{x['object_entity']['type']}", axis=1)
                train_df["subject_type"] = train_df.apply(lambda x: f"{x['subject_entity']['type']}", axis=1)
                train_df.sort_values(by="subject_type", inplace=True)
                train_df.drop("subject_type", axis=1, inplace=True)

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
            shuffle=True if self.cfg["model_class"] != "RECENT" else False # 정렬한 데이터를 다시 섞지 않기 위함.
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
