# -*- coding: utf-8 -*-
from googletrans import Translator
from typing import Tuple, List
import multiprocessing as mp
import pandas as pd
import logging
from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('--lang',type=str,default='en')
parser.add_argument('--start',type=int,default=0)
parser.add_argument('--end',type=int,default=32470)
parser.add_argument('--train',type=str,default='./data/train_data.csv')
arg = parser.parse_args()

LANG=arg.lang
START=arg.start
END=arg.end
TRAIN_DATA_PATH=arg.train

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
file_handler = logging.FileHandler(f'backtranslation_{LANG}_{START}.log')
logger.addHandler(file_handler)


def backtranslate(text:str,translator:Translator, lang='en'):
    translated=translator.translate(text=text, src='ko',dest=lang).text
    back_translated=translator.translate(text=translated, src=lang, dest='ko').text
    return back_translated

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

def mask_backtranslate(
        item:pd.Series,
        translator:Translator,
        lang:str='en'
    )->Tuple[str,int,int,int,int]:

    new_item=item.copy()
    text = item['sentence']

    sub = item['subject_entity']['word']
    sub_start = item['subject_entity']['start_idx']
    sub_end = item['subject_entity']['end_idx']

    obj = item['object_entity']['word']
    obj_start = item['object_entity']['start_idx']
    obj_end = item['object_entity']['end_idx']

    # print(sub,obj)
    if sub_start>obj_start:
        if sub_end==len(text)-1:
            text=text[:sub_start]+'@'
        else:
            text=text[:sub_start]+'@'+text[sub_end+1:]
        text=text[:obj_start]+'#'+text[obj_end+1:]
    else:
        if obj_end==len(text)-1:
            text=text[:obj_start]+'#'
        else:
            text=text[:obj_start]+'#'+text[obj_end+1:]
        text=text[:sub_start]+'@'+text[sub_end+1:]
    # print(text)
    post_txt = backtranslate(text,translator,lang)
    # print(text)

    sub_pos = post_txt.find('@')
    obj_pos = post_txt.find('#')
    if sub_pos==-1 or obj_pos==-1:
        if sub in post_txt and obj in post_txt:
            if sub_pos==-1:
                sub_pos=post_txt.find(sub)
            if obj_pos==-1:
                obj_pos=post_txt.find(obj)
        else:
            logger.warning(text)
            logger.warning(post_txt)
            logger.warning('^'.join([sub,obj]))
            return False
        
    if sub_pos<obj_pos:
        post_txt=post_txt.replace('@',sub)
        sub_start=sub_pos
        sub_end=sub_pos+len(sub)-1

        obj_pos = post_txt.find('#')
        post_txt=post_txt.replace('#',obj)
        obj_start=obj_pos
        obj_end=obj_pos+len(obj)-1
    else:
        post_txt=post_txt.replace('#',obj)
        obj_start=obj_pos
        obj_end=obj_pos+len(obj)-1

        sub_pos = post_txt.find('@')
        post_txt=post_txt.replace('@',sub)
        sub_start=sub_pos
        sub_end=sub_pos+len(sub)-1
    # print(post_txt)
    new_item['sentence']=post_txt
    new_item['subject_entity']['start_idx']=sub_start
    new_item['subject_entity']['end_idx']=sub_end
    new_item['object_entity']['start_idx']=obj_start
    new_item['object_entity']['end_idx']=obj_end
    return new_item



if __name__=="__main__":
    START=arg.start
    df = pd.read_csv('./data/train_data.csv')
    # df=df.loc[start_idx:]
    df=preprocessing_dataset(df)
    translator=Translator()
    backtranslated_df=pd.DataFrame(columns=df.columns)
    success=0
    failed=0
    no_relation=0
    try:
        for i in range(START, END):
            if df.loc[i,'label']!='no_relation':
                back_trans_item = mask_backtranslate(df.loc[i], translator, LANG)
                if isinstance(back_trans_item,pd.Series):
                    backtranslated_df.loc[len(backtranslated_df)] = back_trans_item
                    success+=1
                else:
                    logger.warning(i)
                    logger.warning('')
                    failed+=1
            else:
                no_relation+=1
            if i%1000==0:
                print(success,failed,no_relation)
    except Exception as e:
        print(e)
    finally:
        backtranslated_df.to_csv(f'./backtranslated_{LANG}_{START}.csv',mode='a',index=False,header=False)