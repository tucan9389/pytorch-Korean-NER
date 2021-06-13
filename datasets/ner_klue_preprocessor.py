from tqdm import tqdm
from datasets.ner_preprocessor import NERPreprocessor

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