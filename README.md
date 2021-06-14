# pytorch-NER

## TODO

- [x] 한국어
  - [x] KLUE-NER
  - [x] KMOU-NER
  - [x] NAVER-NER
- [ ] 일본어
- [ ] 그 외 언어

## Train

### git clone

```shell
git clone https://github.com/tucan9389/pytorch-NER
cd pytorch-NER
```

### pip install

```shell
pip install -r requirements.txt
```

### python train.py

```shell
python train.py --gpu_device 0 1
# or
python train.py \
    --gpu_device 0 1 \
    --dataset_root_path /home/centos/datasets \
    --dataset "KLUE" \
    --model_name "koelectra-v3" \
    --epoch 300 \
    --bs 64
```

## Inference

> preparing...

## Results

### KLUE

model | entity<br>f1 score | char<br>f1 score
-- | :--: | :--:
koelectra-base-v3 | **0.926** | 0.930
koelectra-base | 0.925 | **0.935**
kcbert-base | 0.890 | 0.904

### KMOU(한국해양대)

model | entity<br>f1 score | char<br>f1 score
-- | :--: | :--:
koelectra-base-v3 | **0.890** |
koelectra-base | 0.885 |
kcbert-base | 0.871 |

### NAVER

model | entity<br>f1 score | char<br>f1 score
-- | :--: | :--:
koelectra-base-v3 |  |
koelectra-base | 0.885 | 0.885
kcbert-base | 0.885 | 0.874

## Reference

- https://github.com/eagle705/pytorch-bert-crf-ner
- https://github.com/Beomi/KcBERT-Finetune
- https://github.com/KLUE-benchmark/KLUE
- https://github.com/naver/nlp-challenge
- https://github.com/kmounlp/NER
