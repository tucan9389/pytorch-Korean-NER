import os
from tqdm import tqdm
from datasets.ner_preprocessor import NERPreprocessor

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