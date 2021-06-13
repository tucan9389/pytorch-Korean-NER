from tqdm import tqdm
from datasets.ner_preprocessor import NERPreprocessor

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