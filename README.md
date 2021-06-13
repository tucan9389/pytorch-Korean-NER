# pytorch-NER

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

model | entity<br>f1 score | char<br>f1 score | colab
-- | :--: | :--: | :--:
koelectra-base-v3 | **0.926** |  | [link](https://colab.research.google.com/drive/1dCsFmOeUZAR7DY6jJNITyG8AeP9RTzdM?usp=sharing)
koelectra-base | 0.925 |  | [link](https://colab.research.google.com/drive/1aKCaI_c_Mg8f0DK4cgOIVEsTRO_0avEp?usp=sharing)
kcbert-base | 0.890 |  | [link](https://colab.research.google.com/drive/1uJxX6pRsi9O13J7agSrf6bpTIfxLOJg1?usp=sharing)

### KMOU(한국해양대)

model | entity<br>f1 score | char<br>f1 score | colab
-- | :--: | :--: | :--:
koelectra-base-v3 | **0.890** |  | [link](https://colab.research.google.com/drive/1e_35D_WigNDu48mO1MjkRjpb-jCgouxt?usp=sharing)
koelectra-base | 0.885 |  | [link](https://colab.research.google.com/drive/10B0HdxH0HnLz9UidRlztp0CTvN3EtEXc?usp=sharing)
kcbert-base | 0.871 |  | [link](https://colab.research.google.com/drive/1Dg08ZjLu4T1LjwCoAnc0XW-eU2olTPz-?usp=sharing)

### NAVER

model | entity<br>f1 score | char<br>f1 score | colab
-- | :--: | :--: | :--:
koelectra-base-v3 | **0.877** |  | [link](https://colab.research.google.com/drive/1LuiGHDkJnpWyOkN7FUqzCNXWi9fpYvH8?usp=sharing)
koelectra-base | 0.876 |  | [link](https://colab.research.google.com/drive/1wL5al6DPTbP3IceX1I993v-1G3Q8kJwE?usp=sharing)
kcbert-base | 0.863 |  | [link](https://colab.research.google.com/drive/19B4HneG4BUJK_Nac6PAt6SfLB4Q-tzXs?usp=sharing)

## Reference

- https://github.com/eagle705/pytorch-bert-crf-ner
- https://github.com/Beomi/KcBERT-Finetune
- https://github.com/KLUE-benchmark/KLUE
- https://github.com/naver/nlp-challenge
- https://github.com/kmounlp/NER
