# Pytorch Korean NER

## TODO

- [x] 한국어
  - [x] KLUE-NER
  - [x] KMOU-NER
  - [x] NAVER-NER
- [ ] 일본어
- [ ] 그 외 언어

## What is NER?

NER(Named Entity Recognition: 개체명인식)은 주어진 문장에대해 미리 정의한 개체명 카테고리를 찾고, 주어진 문장에서 해당 개체명의 위치를 찾는 task입니다. 

아래와 같이 입력 문장이 주어졌을때, 
```
토코투칸은 세상에서 가장 아름다운 새로, 중부 및 동부 남아메리카에 서식한다.
```

label된 데이터는 아래와 같을 수 있으며 NER 모델 추론결과이기도 합니다. `토코투칸`에 `ANIMAL`로, `중부 및 동부 남아메리카`에 `LOCATION`으로 레이블링된 데이터입니다. 보통은 `ANIMAL` 대신 `ANI`로, `LOCATION` 대신 `LOC`으로 표기하기도 합니다.
```
<토코투칸:ANIMAL>은 세상에서 가장 아름다운 새로, <중부 및 동부 남아메리카:LOCATION>에 서식한다.
```

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
