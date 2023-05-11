import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    DataCollatorForLanguageModeling, 
    AutoModelForMaskedLM, 
    TrainingArguments, Trainer, 
    PreTrainedModel,
    MODEL_FOR_MASKED_LM_MAPPING
)
from datasets import Dataset
import random
from typing import List, Union
import pandas as pd
import yaml, json
import re
import huggingface_hub
from utils import get_result_name
from functools import partial
import wandb
import os

# https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling

def preprocess_function(examples,tokenizer):
    sentence :str = examples["sentence"]
    return tokenizer(sentence)


def group_texts(examples, block_size:int):
    """텍스트를 모두 한줄로 만든 후, `block_size`만큼 나누는 함수

    Args:
        examples (_type_): _description_
        block_size (int): _description_. Defaults to 128.

    Returns:
        _type_: _description_
    """
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def MaskedLanguageModeling(cfg,run_name):
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'])

    df = pd.read_csv(cfg['data_path'])
    dataset = Dataset.from_pandas(df)
    dataset = dataset.flatten()
    dataset = dataset.train_test_split(test_size = cfg['test_size'])
    print(dataset)

    dataset = dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=dataset['train'].column_names,
    )
    print(dataset)

    # dataset = dataset.map(
    #     partial(group_texts, block_size=cfg['block_size']), 
    #     batched=True, 
    #     num_proc=4
    # )
    # print(dataset)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True,
        mlm_probability=cfg['mlm_probability']
    )

    model :PreTrainedModel = AutoModelForMaskedLM.from_pretrained(cfg['model'])

    training_args = TrainingArguments(
        seed=cfg['seed'],
        dataloader_num_workers=cfg['num_workers'],

        output_dir="results_mlm",
        logging_strategy='epoch',
        evaluation_strategy="steps",
        eval_steps=cfg['save_steps'],
        save_strategy='steps',
        save_steps=cfg['save_steps'],
        save_total_limit=1,

        learning_rate=cfg['lr'],
        per_device_train_batch_size=cfg['train_batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'],
        num_train_epochs=cfg['epoch'],
        weight_decay=cfg['weight_decay'],

        report_to='wandb',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except Exception as e:
        print("Error during training:", e)
    finally:
        trainer.save_model(os.path.join(cfg['save_dir'],run_name))
        wandb.finish()


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    run_name = get_result_name()
    wandb.init(
        entity=cfg['entity'],
        project=cfg['project'],
        name=run_name,
    )

    MaskedLanguageModeling(cfg, run_name)


if __name__=="__main__":
    train("config_mlm.yaml")