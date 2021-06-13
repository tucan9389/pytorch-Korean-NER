import os, re
import argparse

from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
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

from torch.utils.tensorboard import SummaryWriter

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

def main(gpu, ngpus_per_node, args):
    print("gpu:", gpu)
    print("ngpus_per_node:", ngpus_per_node)
    print("args:", args)

    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
        
    args.rank = args.rank * ngpus_per_node + gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)

    args.bs = int(args.bs / ngpus_per_node)
    args.num_workers = int(args.num_workers / ngpus_per_node)

    # dataset config
    dataset_name    = args.dataset
    dataset_root_path = args.dataset_root_path

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

    echo_num        = args.echo_num

    writer = None
    if args.rank == 0:
        # tensorboard
        log_dir = "logs"
        exp_log_dir = os.path.join(log_dir, f"{dataset_name}-{model_name}-lr{learning_rate}-ae{adam_epsilon}-bs{batch_size}-ep{total_epoch_num}")
        os.makedirs(exp_log_dir, exist_ok=True)
        writer = SummaryWriter(exp_log_dir)
    
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        url = tb.launch()
        print(f"Open Tensorboard at {log_dir}:", url)

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
    os.makedirs(dataset_root_path, exist_ok=True)
    dataset_train, dataset_valid = get_dataset(dataset_name, dataset_root_path, tokenizer, max_seq_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.bs, 
                                  shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                  sampler=train_sampler, drop_last=True)
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
    model.cuda(args.gpu)
    model_dp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    device = args.gpu
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)

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
        with tqdm(dataloader_train, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                model.zero_grad()

                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]
                }
                
                outputs = model_dp(**inputs)

                loss = outputs[0]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model_dp.parameters(), max_grad_norm)

                optimizer.step()
                # scheduler.step()
                # model.zero_grad()

                total_loss += loss.mean().item()
                echo_loss += loss.mean().item()

                tepoch.set_postfix(loss=loss.mean().item())
                

        if args.rank == 0:
            loss = total_loss / total_step_per_epoch
            
            
            precision_e, recall_e, f1score_e, \
            precision_c, recall_c, f1score_c = evaluate(dataloader_valid, model_dp, device, tokenizer, id2label)
            best_f1score_e = max(best_f1score_e, f1score_e)
            best_f1score_c = max(best_f1score_c, f1score_c)

            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('valid/entity/precision', precision_e, epoch)
            writer.add_scalar('valid/entity/recall', recall_e, epoch)
            writer.add_scalar('valid/entity/f1-score', f1score_e, epoch)
            writer.add_scalar('valid/char/precision', precision_c, epoch)
            writer.add_scalar('valid/char/recall', recall_c, epoch)
            writer.add_scalar('valid/char/f1-score', f1score_c, epoch)

            print(f"[epoch:{epoch+1}/{total_epoch_num}] entity train-loss:{loss:.3f}, valid-precision:{precision_e:.3f}, valid-recall:{recall_e:.3f}, valid-f1score:{f1score_e:.3f}, best-valid-f1score:{best_f1score_e:.3f}")
            print(f"[epoch:{epoch+1}/{total_epoch_num}] char   train-loss:{loss:.3f}, valid-precision:{precision_c:.3f}, valid-recall:{recall_c:.3f}, valid-f1score:{f1score_c:.3f}, best-valid-f1score:{best_f1score_c:.3f}")
            print()
        if args.distributed:
            dist.barrier()
    
    if args.rank == 0:
        writer.add_scalar('valid/f1-score/entity-best', best_f1score_e, 0)
        writer.add_scalar('valid/f1-score/char-best', best_f1score_c, 0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    parser.add_argument('--dataset', default='KLUE', choices=["KLUE", "KMOU", "NAVER"])
    parser.add_argument('--dataset_root_path', required=True)
    parser.add_argument('--model_name', default='koelectra', choices=["koelectra-v3", "koelectra", "kcbert"])
    parser.add_argument('--max_seq_len', default=50, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--ae', default=1e-8, type=float)
    parser.add_argument('--echo_num', default=5, type=int)

    args = parser.parse_args()

    torch.cuda.empty_cache()

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    # distributed env
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    print("ngpus_per_node:", ngpus_per_node)

    mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
