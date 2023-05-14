import pickle
import pandas as pd
import numpy as np
import sklearn
from datetime import datetime, timezone, timedelta
from typing import List


def load_data(data_dir: str) -> pd.DataFrame:
    """csv 파일을 경로에 맞게 불러 옵니다."""
    df = pd.read_csv(data_dir)

    return df


def label_to_num(label: List) -> List:
    """문자열 class label을 숫자로 변환합니다."""
    if label[0] == 100:  # test_data를 받은 경우
        return label
    num_label = []
    with open("./src/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label: List[int]) -> List[str]:
    """숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다."""
    if len(set(np.array(label, dtype=np.int64).reshape(-1))) <= 2: # binary-label의 경우 no-relation or others로 분류
        label = np.array(label, dtype=np.int64).reshape(-1)
        label = np.where(label<0.5, 'no_relation', 'others')
        return label.tolist()
    
    origin_label = []
    with open("./src/dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def preprocessing_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for sub, obj in zip(df["subject_entity"], df["object_entity"]):
        sub = eval(sub)
        obj = eval(obj)
        subject_entity.append(sub)
        object_entity.append(obj)
    out_dataset = pd.DataFrame(
        {
            "id": df["id"],
            "sentence": df["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": df["label"],
        }
    )

    return out_dataset


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(
            labels,
            preds,
            average="micro",
            labels=label_indices,
        )
        * 100.0
    )


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]
    probs = np.array(probs)
    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def get_result_name() -> str:
    """한국 시간으로 result 이름을 반환합니다."""
    now = datetime.now(tz=timezone(timedelta(hours=9)))

    return now.strftime("%m-%d-%H:%M:%S")


def remove_pad_tokens(sentences: List[str], pad_token: str) -> List[str]:
    """pad token만 제거하는 함수입니다."""
    ret = [sentence.replace(" " + pad_token, "") for sentence in sentences]
    return ret

def tokenizer_update(tokenizer, cfg):
    """entity masking을 위해 tokenzier update하는 함수
    """
    input_format = cfg['input_format']
    if input_format in ['default','entity_marker_punct, typed_entity_marker_punct']:
        return tokenizer
    
    df = pd.read_csv(cfg['train_dir'])
    new_tokens = []
    types = []
    if input_format == 'entity_marker':
        new_tokens = ['[E1]','[/E1]','[E2]','[/E2]']
    else:
        for sub, obj in zip(df["subject_entity"], df["object_entity"]):
            sub = eval(sub)
            obj = eval(obj)
            if input_format == 'entity_mask':
                subj_type = '[SUBJ-{}]'.format(sub['type'])
                obj_type = '[OBJ-{}]'.format(obj['type'])
                types = [subj_type, obj_type]
            elif input_format == 'typed_entity_marker':
                subj_type1 = '[S:{}]'.format(sub['type'])
                subj_type2 = '[/S:{}]'.format(sub['type'])
                obj_type1 = '[O:{}]'.format(obj['type'])
                obj_type2 = '[/O:{}]'.format(obj['type'])
                types = [subj_type1, subj_type2, obj_type1, obj_type2]
            for token in types:
                if token not in new_tokens:
                    new_tokens.append(token)

    tokenizer.add_tokens(new_tokens, special_tokens=True)
    return tokenizer

def model_freeze(model, keys_to_remain:list = []):
    for name, param in model.named_parameters():
        param.requires_grad = False
        for key in keys_to_remain:
            if key in name:
                param.requires_grad =True
    return model