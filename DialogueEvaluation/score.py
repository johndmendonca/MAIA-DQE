from argparse import ArgumentParser
import transformers
from transformers import (
    Trainer,
    AutoConfig,
    AutoModelWithHeads,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    XLMRobertaForMaskedLM,
)
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import logging
import torch
import pandas as pd
from datetime import datetime
import json
from scipy import stats
from pathlib import Path
from sklearn.model_selection import train_test_split
import re


def preprocess_ctxres(examples):
    args = (examples["ctx_src"], examples["res_src"])
    result = tokenizer(
        *args, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    return result


def preprocess_ctx(examples):
    args = (examples["res_src"],)
    result = tokenizer(
        *args, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    return result


def main():
    parser = ArgumentParser(description='Obtain scores')
    parser.add_argument('vsp', type=str, help='vsp model path')
    parser.add_argument('nsp', type=str, help='nsp model path')
    parser.add_argument('eng', type=str, help='eng model path')
    parser.add_argument('eval', type=str, help='evaluation set')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_args = TrainingArguments(
        per_device_eval_batch_size=32,
        do_predict=True,
        output_dir="out",
        label_names=["labels"],
    )

    config = AutoConfig.from_pretrained(
        'xlm-roberta-large',
        num_labels=1,
    )
    special_token_dict = {
        "speaker1_token": "<speaker1>",
        "speaker2_token": "<speaker2>",
    }

    tokenizer = AutoTokenizer.from_pretrained(
        'xlm-roberta-large',
        use_fast=True,
    )
    tokenizer.add_tokens(list(special_token_dict.values()))

    model = AutoModelWithHeads.from_pretrained(
        args.vsp,
        config=args.vsp,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    ds = load_dataset(
        "csv",
        data_files={"test": args.eval},
        download_mode="force_redownload",
    )
    ds = ds.map(preprocess_ctx, batched=True)
    preds_vsp, _, _ = trainer.predict(
        test_dataset=ds["test"]
    )
    with open(f"vsp.json", "w") as outfile:
        json.dump(preds_vsp.squeeze().tolist(), outfile)

    model = AutoModelWithHeads.from_pretrained(
        args.nsp,
        config=args.nsp,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    ds = load_dataset(
        "csv",
        data_files={"test": args.eval},
        download_mode="force_redownload",
    )
    ds = ds.map(preprocess_ctx, batched=True)
    preds_nsp, _, _ = trainer.predict(
        test_dataset=ds["test"]
    )
    with open(f"vsp.json", "w") as outfile:
        json.dump(preds_nsp.squeeze().tolist(), outfile)
    
    model = AutoModelWithHeads.from_pretrained(
        args.eng,
        config=args.eng,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    ds = load_dataset(
        "csv",
        data_files={"test": args.eval},
        download_mode="force_redownload",
    )
    ds = ds.map(preprocess_ctx, batched=True)
    preds_eng, _, _ = trainer.predict(
        test_dataset=ds["test"]
    )
    with open(f"eng.json", "w") as outfile:
        json.dump(preds_eng.squeeze().tolist(), outfile)
    
    # Evaluation
    #labels = pd.read_csv("data/MAIA/de_client_01_turn.csv").templated
    #print(balanced_accuracy_score(np.array(labels)>0, np.array(data["APPROPRIATENESS"])>0.5, ))
    #print(balanced_accuracy_score(np.array(labels)>0, np.array(data["RELEVANCE"])>0.5, ))
    #print(balanced_accuracy_score(np.array(labels)>0, np.array(data["CONTENT_RICHNESS"])>0.5, ))
    #print(balanced_accuracy_score(np.array(labels)>0, np.array(data["GRAMMATICAL_CORRECTNESS"])>0.5, ))