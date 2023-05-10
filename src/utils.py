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


def label_to_num(label: List[str]) -> List[int]:
    """문자열 class label을 숫자로 변환합니다."""
    num_label = []
    with open("./src/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label: List[int]) -> List[str]:
    """숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다."""
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
        subject_entity.append(sub["word"])
        object_entity.append(obj["word"])
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


def remove_pad_tokens(sentences: list[str], pad_token: str) -> list[str]:
    """pad token만 제거하는 함수입니다."""
    ret = [sentence.replace(" " + pad_token, "") for sentence in sentences]
    return ret
