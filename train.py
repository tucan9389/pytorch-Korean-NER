import os, re
import argparse

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from fastprogress.fastprogress import master_bar, progress_bar
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers import (
    BertConfig, BertForTokenClassification, BertTokenizer, # for KcBERT
    ElectraConfig, ElectraTokenizer, ElectraForTokenClassification,
    AutoConfig, AutoTokenizer, AutoModelForTokenClassification, 
)
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from datasets.dataset_provider import get_dataset
from models.model_provider import get_model

def evaluate(dataloader_valid, model, device, tokenizer, id2label, echo_num=40):

    total_step_per_epoch = len(dataloader_valid)
    total_loss = 0.0
    in_token_ids = None
    preds = None
    out_label_ids = None

    model.eval()
    for step, batch in enumerate(dataloader_valid):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }

            outputs = model(**inputs)

            loss, logits = outputs[:2]
            total_loss += loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            in_token_ids = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            in_token_ids = np.append(in_token_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    loss = total_loss / total_step_per_epoch
    preds = np.argmax(preds, axis=2)

    gt_token_label_list = [[] for _ in range(out_label_ids.shape[0])]
    pred_token_label_list = [[] for _ in range(out_label_ids.shape[0])]
    gt_char_label_list = [[] for _ in range(out_label_ids.shape[0])]
    pred_char_label_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    for i in range(out_label_ids.shape[0]): # sentence
        token_ids = in_token_ids[i]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        for j in range(out_label_ids.shape[1]): # token
            if out_label_ids[i, j] == pad_token_label_id: continue

            gt_label_id = out_label_ids[i][j]
            gt_label = id2label[gt_label_id]
            gt_token_label_list[i].append(gt_label)
            pred_label_id = preds[i][j]
            pred_label = id2label[pred_label_id]
            pred_token_label_list[i].append(pred_label)

            token = tokens[j]
            token = token.replace("##", "")
            if token[0] == '[' and token[-1] == ']':
                gt_char_label_list[i].append(gt_label)
                pred_char_label_list[i].append(pred_label)
            else:
                gt_char_label_list[i] += [gt_label]*len(token)
                pred_char_label_list[i] += [pred_label]*len(token)

    result = classification_report(gt_token_label_list, pred_token_label_list)
    print("[entity f1 score]")
    print(result)

    result = classification_report(gt_char_label_list, pred_char_label_list)
    print("[char f1 score]")
    print(result)
    
    return precision_score(gt_token_label_list, pred_token_label_list), \
    recall_score(gt_token_label_list, pred_token_label_list), \
    f1_score(gt_token_label_list, pred_token_label_list), \
    precision_score(gt_char_label_list, pred_char_label_list), \
    recall_score(gt_char_label_list, pred_char_label_list), \
    f1_score(gt_char_label_list, pred_char_label_list)
    # results.update(result)


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--dataset', default='KLUE', choices=["KLUE", "KMOU", "NAVER"])
    my_parser.add_argument('--model_name', default='koelectra', choices=["koelectra-v3", "koelectra", "kcbert"])
    my_parser.add_argument('--max_seq_len', default=50, type=int)
    my_parser.add_argument('--epoch', default=300, type=int)
    my_parser.add_argument('--bs', default=128, type=int)
    my_parser.add_argument('--lr', default=1e-3, type=float)
    my_parser.add_argument('--ae', default=1e-8, type=float)
    my_parser.add_argument('--echo_num', default=5, type=int)

    args = my_parser.parse_args()

    # dataset config
    dataset_name    = args.dataset

    # model config
    model_name      = args.model_name
    max_seq_len     = args.max_seq_len

    # training config
    batch_size      = args.bs
    total_epoch_num = args.epoch
    weight_decay    = 0.0
    learning_rate   = args.lr  # 8e-5  # 3e-4
    adam_epsilon    = args.ae
    warmup_steps    = 0
    max_grad_norm   = 1.0

    echo_num        = int(args.echo_num)

    # get model
    ModelConfig, \
    Tokenizer, \
    Model, \
    pretraine_name = get_model(model_name=model_name)

    # get tokenizer
    tokenizer = Tokenizer.from_pretrained(pretraine_name, do_lower_case=False)

    print(tokenizer.tokenize("안녕하세요 저는 곽도영입니다.👍"))  # ['안녕', '##하세요', '저는', '곽', '##도', '##영', '##입니다', '.', '👍']
    print(tokenizer.tokenize("이미 수상자(2000년 김대중 전 대통령)를 배출한 데다"))  # ['이미', '수상', '##자', '(', '2000', '##년', '김대중', '전', '대통령', ')', '를', '배출', '##한', '데', '##다']
    print(tokenizer.tokenize("한반도의 운명은"))  # ['한반도의', '운명', '##은'], "<한반도:LOC>의운명은"

    # get dataset
    dataset_train, dataset_valid = get_dataset(dataset_name, tokenizer, max_seq_len)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True) # , collate_fn=??)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, drop_last=False)

    t_total = len(dataloader_train) // total_epoch_num

    labels    = dataset_train.labels
    id2label  = {i: label for i, label in enumerate(labels)}
    label2id  = {label: i for i, label in enumerate(labels)}
    print("len(labels):", len(labels))
    print("labels:", labels)
    print("id2label:", id2label)
    print("label2id:", label2id)
    print()

    config    = ModelConfig.from_pretrained(pretraine_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
    model     = Model.from_pretrained(pretraine_name, config=config)

    # GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    if Model == BertForTokenClassification:
        optimizer_grouped_parameters = [
            {'params': model.bert.parameters(), 'lr': learning_rate / 100 },
            {'params': model.classifier.parameters(), 'lr': learning_rate }
        ]
    elif Model == ElectraForTokenClassification:
        optimizer_grouped_parameters = [
            {'params': model.electra.parameters(), 'lr': learning_rate / 100 },
            {'params': model.classifier.parameters(), 'lr': learning_rate }
        ]
    else:
        assert False, f"{Model} is not supported"

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    # optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # mb = master_bar(range(total_epoch_num))
    total_echo_num = 20
    echo_loss = 0.0
    best_f1score_e = 0.0
    best_f1score_c = 0.0

    total_step_per_epoch = len(dataloader_train)
    for epoch in range(total_epoch_num):
        # epoch_iterator = progress_bar(dataloader_train, parent=mb)

        # train one epoch
        total_step_per_epoch = len(dataloader_train)
        total_loss = 0.0
        echo_loss = 0.0

        model.train()
        for step, batch in enumerate(dataloader_train):
            model.zero_grad()

            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }
            
            outputs = model(**inputs)

            loss = outputs[0]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            # scheduler.step()
            # model.zero_grad()

            total_loss += loss.mean().item()
            echo_loss += loss.mean().item()

            if (step+1) % echo_num == 0:
                print(f" >> [epoch:{epoch+1}/{total_epoch_num}][step:{step+1}/{total_step_per_epoch}] train-loss:{echo_loss/float(echo_num):.3f}")
                echo_loss = 0.0

        loss = total_loss / total_step_per_epoch
        
        
        precision_e, recall_e, f1score_e, \
        precision_c, recall_c, f1score_c = evaluate(dataloader_valid, model, device, tokenizer, id2label)
        best_f1score_e = max(best_f1score_e, f1score_e)
        best_f1score_c = max(best_f1score_c, f1score_c)

        print(f"[epoch:{epoch+1}/{total_epoch_num}] entity train-loss:{loss:.3f}, valid-precision:{precision_e:.3f}, valid-recall:{recall_e:.3f}, valid-f1score:{f1score_e:.3f}, best-valid-f1score:{best_f1score_e:.3f}")
        print(f"[epoch:{epoch+1}/{total_epoch_num}] char   train-loss:{loss:.3f}, valid-precision:{precision_c:.3f}, valid-recall:{recall_c:.3f}, valid-f1score:{f1score_c:.3f}, best-valid-f1score:{best_f1score_c:.3f}")
        print()
