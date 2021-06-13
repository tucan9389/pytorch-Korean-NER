from datasets.ner_naver_preprocessor import NAVERNERPreprocessor
from datasets.ner_klue_preprocessor import KLUENERPreprocessor
from datasets.ner_kmou_preprocessor import KMOUNERPreprocessor
from datasets.ner_dataset import NERDataset

def get_dataset(dataset_name, tokenizer, max_seq_len=50):
    if dataset_name == "KLUE":
        dataset_path_train = "/home/mldongseok/datasets/nlp//KLUE/klue_benchmark/klue-ner-v1/klue-ner-v1_train.tsv"
        dataset_path_dev = "/home/mldongseok/datasets/nlp/KLUE/klue_benchmark/klue-ner-v1/klue-ner-v1_dev.tsv"
        sentences = open(dataset_path_train, 'r').read().split("\n## ")
        del sentences[:5]
    elif dataset_name == "KMOU":
        dataset_path_train  = "/home/mldongseok/datasets/nlp/pytorch-bert-crf-ner/data_in/NER-master/말뭉치 - 형태소_개체명"
        dataset_path_dev  = "/home/mldongseok/datasets/nlp/pytorch-bert-crf-ner/data_in/NER-master/validation_set"
    elif dataset_name == "NAVER":
        dataset_path_train  = "/home/mldongseok/datasets/nlp/KcBERT-Finetune/data/naver-ner/train.tsv"
        dataset_path_dev  = "/home/mldongseok/datasets/nlp/KcBERT-Finetune/data/naver-ner/test.tsv"

    if dataset_name == "KLUE":
        Processor = KLUENERPreprocessor
    elif dataset_name == "KMOU":
        Processor = KMOUNERPreprocessor
    elif dataset_name == "NAVER":
        Processor = NAVERNERPreprocessor
    else:
        Processor = None
        assert False, f"{dataset_name} ner dataset is not supported"

    dataset_train = NERDataset(dataset_path_train, tokenizer=tokenizer, max_seq_len=max_seq_len, Preprocessor=Processor)
    dataset_valid = NERDataset(dataset_path_dev , tokenizer=tokenizer, max_seq_len=max_seq_len, labels=dataset_train.labels, Preprocessor=Processor)

    return dataset_train, dataset_valid