import os, re

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from fastprogress.fastprogress import master_bar, progress_bar

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import numpy as np

from transformers import (
    BertConfig, BertForTokenClassification, BertTokenizer, # for KcBERT
    ElectraConfig, ElectraTokenizer, ElectraForTokenClassification,
    AutoConfig, AutoTokenizer, AutoModelForTokenClassification, 
)

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

dataset_name = "NAVER" # NAVER NER
# dataset_name = "KMOU" # KOREA MARITIME & OCEAN UNIVERSITY
# dataset_name = "KLUE"  # Korean Language Understanding Evaluation

if dataset_name == "KLUE":
    dataset_path_train = "/home/centos/datasets/nlp/KLUE/klue_benchmark/klue-ner-v1/klue-ner-v1_train.tsv"
    dataset_path_dev = "/home/centos/datasets/nlp/KLUE/klue_benchmark/klue-ner-v1/klue-ner-v1_dev.tsv"
    sentences = open(dataset_path_train, 'r').read().split("\n## ")
    del sentences[:5]
elif dataset_name == "KMOU":
    dataset_path_train  = "/home/centos/datasets/nlp/pytorch-bert-crf-ner/data_in/NER-master/말뭉치 - 형태소_개체명"
    dataset_path_dev  = "/home/centos/datasets/nlp/pytorch-bert-crf-ner/data_in/NER-master/validation_set"
elif dataset_name == "NAVER":
    dataset_path_train  = "/home/centos/datasets/nlp/KcBERT-Finetune/data/naver-ner/train.tsv"
    dataset_path_dev  = "/home/centos/datasets/nlp/KcBERT-Finetune/data/naver-ner/test.tsv"

class NERPreprocessor(object):
    @staticmethod
    def preprocess(tokenizer, splited_texts_list, splited_tags_list, label_map, max_seq_len, pad_token_label_id=-100):  # bert의 최대 maxlen을 사용하는것을 
        list_of_token_ids = []
        list_of_attension_mask = []
        list_of_label_ids = []

        before_text = ' '
        for splited_texts, splited_tags in tqdm(zip(splited_texts_list, splited_tags_list)):
            
            tokens_for_a_sentence = []
            tags_for_a_sentence = []
            for text, tag in zip(splited_texts, splited_tags):
                if text == ' ':
                    continue
                if before_text != ' ':
                    tokens = tokenizer.tokenize('##' + text)
                else:
                    tokens = tokenizer.tokenize(text)
                tokens_for_a_sentence += tokens
                tags_for_a_sentence += list(map(lambda x: 'O' if tag == 'O' else (f'B-{tag}' if x==0 else f'I-{tag}'), range(len(tokens))))

            
            if len(tokens_for_a_sentence) > max_seq_len - 2:
                tokens_for_a_sentence = tokens_for_a_sentence[:max_seq_len - 2]
                tags_for_a_sentence = tags_for_a_sentence[:max_seq_len - 2]
            
            tokens_for_a_sentence = [tokenizer.cls_token] + tokens_for_a_sentence + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens_for_a_sentence)
            padding_length = max_seq_len - len(token_ids)

            attension_mask = [1]*len(token_ids) + [0]*padding_length
            token_ids += [tokenizer.pad_token_id]*padding_length

            label_ids = list(map(lambda x: label_map[x], tags_for_a_sentence))
            label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
            label_ids += [pad_token_label_id]*padding_length

            list_of_token_ids.append(token_ids)
            list_of_attension_mask.append(attension_mask)
            list_of_label_ids.append(label_ids)
        
        return list_of_token_ids, \
            list_of_attension_mask, \
            list_of_label_ids

class NAVERNERPreprocessor(NERPreprocessor):
    """NAVER NER 데이터셋 전처리"""

    @staticmethod
    def get_split_texts_and_tags_from_dataset_file(dataset_path: str):
        sentences = open(dataset_path, 'r').read().split("\n")

        splited_texts_list = []
        splited_tags_list = []
        labels = ['O']
        for sentence in tqdm(sentences):
            current_tag = 'O'
            splited_texts = []
            splited_tags = []

            if len(sentence.split("\t")) != 2:
                continue
            texts, tags = sentence.split("\t")
            texts_len = len(texts)
            for idx, (text, tag) in enumerate(zip(texts.split(" "), tags.split(" "))):
                text_len = len(text)
                for i in range(text_len):
                    c = text[i]
                    t = tag if len(tag.split('-')) == 1 else tag.split('-')[0]
                    splited_texts.append(c)
                    splited_tags.append(t)

                if idx + 1 != texts_len:
                    splited_texts.append(' ')
                    splited_tags.append('O')

            
            for label in splited_tags:
                if label == 'O':
                    if label not in labels:
                        labels.append(label)
                else:
                    if f'B-{label}' not in labels:
                        labels.append(f'B-{label}')
                    if f'I-{label}' not in labels:
                        labels.append(f'I-{label}')

            splited_texts_list.append(splited_texts)
            splited_tags_list.append(splited_tags)
        
        return splited_texts_list, splited_tags_list, labels

class KMOUNERPreprocessor(NERPreprocessor):
    """KMOU NER 데이터셋 전처리"""
    @staticmethod
    def get_split_texts_and_tags_from_dataset_file(dataset_path: str):
        txt_filenames = os.listdir(dataset_path)
        txt_filenames = list(filter(lambda filename: filename.endswith(".txt"), txt_filenames))
        txt_filepaths = list(map(lambda filename: os.path.join(dataset_path, filename), txt_filenames))

        total_source_text_list = []
        total_target_text_list = []
        for txt_filepath in txt_filepaths:
            source_text_list = []
            target_text_list = []

            f = open(txt_filepath, 'r')
            txt = f.read()
            sentenses = txt.split("\n\n")
            for sentense in sentenses:
                lines = sentense.split("\n")
                if len(lines) < 3:
                    continue
                source_text = lines[1].replace("## ", "")
                target_text = lines[2].replace("## ", "")

                source_text_list.append(source_text)
                target_text_list.append(target_text)

            total_source_text_list += source_text_list
            total_target_text_list += target_text_list
        
        splited_texts_list = []
        splited_tags_list = []
        labels = ['O']
        for source_text, target_text in tqdm(zip(total_source_text_list, total_target_text_list)):
            splited_texts = []
            splited_tags = []
            i, j = 0, 0
            in_tag = False
            tag_starting_index = None
            i_len = len(source_text)
            j_len = len(target_text)
            while i < i_len:
                if not in_tag and source_text[i] == target_text[j]:
                    c = source_text[i]
                    t = 'O'
                elif not in_tag and source_text[i] != target_text[j]:
                    # 개체명 태그 시작지점
                    j += 1
                    assert source_text[i] == target_text[j], "err: source_text[i] != target_text[j]"
                    in_tag = True
                    tag_starting_index = i
                    c = source_text[i]
                    t = None
                elif in_tag and i == i_len-1 and target_text[j+1] == ':':
                    c = source_text[i]
                    tag = target_text[j+2:j+5]
                    for tagging_index in range(tag_starting_index, i):
                        splited_tags[tagging_index] = tag
                    t = tag
                    in_tag = False
                elif in_tag and (source_text[i] != target_text[j] or (target_text[j]==':' and target_text[j+4] == '>')):
                    # 개체명 태그 끝지점
                    if target_text[j] != ':' or target_text[j+4] != '>':
                        print("-" * 30)
                        print("ERROR:")
                        print("  source_text:", source_text)
                        print("  target_text:", target_text)
                        print("  i:", i, "j:", j)
                    assert target_text[j] == ':' and target_text[j+4] == '>', "err: target_text[j] != ':' or target_text[j+4] != '>'"
                    tag = target_text[j+1:j+4]
                    for tagging_index in range(tag_starting_index, i):
                        splited_tags[tagging_index] = tag
                    c = None
                    t = None
                    j += 4
                    i -= 1
                    in_tag = False
                elif in_tag and source_text[i] == target_text[j]:
                    c = source_text[i]
                    t = None
                else:
                    print("-" * 30)
                    print("ERROR:")
                    print("  source_text:", source_text)
                    print("  target_text:", target_text)
                    print("  i:", i, "j:", j)
                    assert False, "something is wrong"
                        
                if c is not None:
                    splited_texts.append(c)
                    splited_tags.append(t)

                i += 1
                j += 1

            # for debugging
            if None in splited_tags:
                print("-" * 30)
                print("ERROR:")
                print("  source_text:", source_text)
                print("  target_text:", target_text)
                for c, t in zip(splited_texts, splited_tags):
                    print(" >>", c, t)
        
            for label in splited_tags:
                if label == 'O':
                    if label not in labels:
                        labels.append(label)
                else:
                    if f'B-{label}' not in labels:
                        labels.append(f'B-{label}')
                    if f'I-{label}' not in labels:
                        labels.append(f'I-{label}')

            splited_texts_list.append(splited_texts)
            splited_tags_list.append(splited_tags)

        return splited_texts_list, splited_tags_list, labels

class KLUENERPreprocessor(NERPreprocessor):
    """KLUE NER 데이터셋 전처리"""

    @staticmethod
    def get_split_texts_and_tags_from_dataset_file(dataset_path: str):
        sentences = open(dataset_path, 'r').read().split("\n## ")
        del sentences[:5]

        # sentences = sentences[:16]  # for debug

        splited_texts_list = []
        splited_tags_list = []
        labels = ['O']
        for sentence in tqdm(sentences):
            current_tag = 'O'
            splited_texts = []
            splited_tags = []
            for idx, line in enumerate(sentence.split("\n")):
                if idx == 0 or len(line.split("\t")) < 2:
                    continue
                
                c, t = line.split("\t")
                if '-' in t:
                    target_tag = t.split("-")[-1]
                else:
                    target_tag = t

                if len(splited_texts) == 0 or target_tag != current_tag:
                    splited_texts.append(c)
                    current_tag = target_tag
                    splited_tags.append(target_tag)
                else:
                    splited_texts[-1] += c
        
            # idx = 0
            # while idx < len(splited_tags):
            #     if splited_texts[idx] == ' ':
            #         del splited_texts[idx]
            #         del splited_tags[idx]
            #         continue
            #     idx += 1
            
            
            for label in splited_tags:
                if label == 'O':
                    if label not in labels:
                        labels.append(label)
                else:
                    if f'B-{label}' not in labels:
                        labels.append(f'B-{label}')
                    if f'I-{label}' not in labels:
                        labels.append(f'I-{label}')

            splited_texts_list.append(splited_texts)
            splited_tags_list.append(splited_tags)
        
        return splited_texts_list, splited_tags_list, labels

    
class NERDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, tokenizer, max_seq_len, labels=None, Preprocessor=None):
        assert os.path.exists(dataset_path)
        self.dataset_path = dataset_path
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len

        source_text_list, target_text_list, new_labels = Preprocessor.get_split_texts_and_tags_from_dataset_file(dataset_path=dataset_path)
        assert len(source_text_list) == len(target_text_list), "Err: Number of source and target is different"
        print(f"{len(source_text_list)}개 문장을 불러왔습니다.")
        labels = labels if labels is not None else new_labels

        # labels = KMAOUNERPreprocessor.create_labels(total_target_text_list=target_text_list)
        label_map = {label: i for i, label in enumerate(labels)}
        print(label_map)

        list_of_token_ids, \
        list_of_attension_mask, \
        list_of_label_ids = Preprocessor.preprocess(tokenizer, source_text_list, target_text_list, label_map, max_seq_len)

        self.list_of_token_ids      = list_of_token_ids
        self.list_of_attension_mask = list_of_attension_mask
        self.list_of_label_ids      = list_of_label_ids
        self.labels                 = labels

    def __len__(self):
        return len(self.list_of_token_ids)

    def __getitem__(self, data_idx: int):
        return torch.Tensor(self.list_of_token_ids[data_idx]).long(), \
            torch.Tensor(self.list_of_attension_mask[data_idx]).long(), \
            torch.Tensor(self.list_of_label_ids[data_idx]).long()

if dataset_name == "KLUE":
    Processor = KLUENERPreprocessor
elif dataset_name == "KMOU":
    Processor = KMOUNERPreprocessor
elif dataset_name == "NAVER":
    Processor = NAVERNERPreprocessor
else:
    Processor = None
    assert False, f"{dataset_name} ner dataset is not supported"

def get_model(model_name="beomi/kcbert-base"):
    ModelConfig = None
    Tokenizer = None
    Model = None
    if model_name == "beomi/kcbert-base":
        ModelConfig = BertConfig
        Tokenizer   = BertTokenizer
        Model       = BertForTokenClassification
    elif model_name == "monologg/koelectra-base-discriminator" or model_name == "monologg/koelectra-base-v3-discriminator":
        ModelConfig = ElectraConfig
        Tokenizer   = ElectraTokenizer
        Model       = ElectraForTokenClassification
    elif model_name == "monologg/kobert":
        ModelConfig = AutoConfig
        Tokenizer   = AutoTokenizer
        Model       = AutoModelForTokenClassification
    return ModelConfig, Tokenizer, Model

"""config 파일 안에 들어갈 내용"""
# dataset_train_path  = "pytorch-bert-crf-ner/data_in/NER-master/말뭉치 - 형태소_개체명"
model_name = "monologg/koelectra-base-v3-discriminator"
ModelConfig, \
Tokenizer, \
Model = get_model(model_name=model_name)
max_seq_len = 50    

tokenizer = Tokenizer.from_pretrained(model_name, do_lower_case=False)

# !ls "pytorch-bert-crf-ner/data_in/NER-master/말뭉치 - 형태소_개체명"
dataset_train = NERDataset(dataset_path_train, tokenizer=tokenizer, max_seq_len=max_seq_len, Preprocessor=NAVERNERPreprocessor)
# dataset_valid = KLUENERDataset(dataset_path_train, tokenizer=tokenizer, max_seq_len=max_seq_len, labels=dataset_train.labels)
dataset_valid = NERDataset(dataset_path_dev , tokenizer=tokenizer, max_seq_len=max_seq_len, labels=dataset_train.labels, Preprocessor=NAVERNERPreprocessor)

labels    = dataset_train.labels
id2label  = {i: label for i, label in enumerate(labels)}
label2id  = {label: i for i, label in enumerate(labels)}
print("len(labels):", len(labels))
print("labels:", labels)
print("id2label:", id2label)
print("label2id:", label2id)
print()

config    = ModelConfig.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
model     = Model.from_pretrained(model_name, config=config)

print(tokenizer.tokenize("안녕하세요 저는 곽도영입니다.👍"))  # ['안녕', '##하세요', '저는', '곽', '##도', '##영', '##입니다', '.', '👍']
print(tokenizer.tokenize("이미 수상자(2000년 김대중 전 대통령)를 배출한 데다"))  # ['이미', '수상', '##자', '(', '2000', '##년', '김대중', '전', '대통령', ')', '를', '배출', '##한', '데', '##다']

print(tokenizer.tokenize("한반도의 운명은"))  # ['한반도의', '운명', '##은'], "<한반도:LOC>의운명은"

# GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

batch_size = 64

# train_sampler = RandomSampler(dataset_train)  # 
# shuffle=True, drop_last=True -> 1,2,3,4,5,6, minibatch:4,     4,3,5,1,6,2 => 4,3,5,1(6,2)
# collate_fn -> mini batch tensor를 만듦, list of tensor로 만
# maxlen 
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True) # , collate_fn=??)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, drop_last=False)

total_epoch_num = 300
weight_decay    = 0.0
learning_rate   = 1e-3  # 8e-5  # 3e-4
adam_epsilon    = 1e-8
warmup_steps    = 0
max_grad_norm   = 1.0
t_total         = len(dataloader_train) // total_epoch_num

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

# mb = master_bar(range(total_epoch_num))
total_echo_num = 20
echo_loss = 0.0
best_f1score_e = 0.0
best_f1score_c = 0.0

total_step_per_epoch = len(dataloader_train)
for epoch in range(total_epoch_num):
    # epoch_iterator = progress_bar(dataloader_train, parent=mb)

    # train one epoch
    echo_num=40
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
