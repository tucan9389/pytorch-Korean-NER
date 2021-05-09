import os, json, re

from torch.utils.data import Dataset
import torch


class NERDataset(Dataset):

    _default_tag_to_index = {"[CLS]": 0, "[SEP]": 1, "[PAD]": 2, "[MASK]": 3, "O": 4}

    def __init__(self, dataset_path: str, caching_tag_dict_path: str=None):
        self.dataset_path = dataset_path

        # load dataset from file system
        source_text_list, target_text_list = self._load_dataset_from_files(dataset_path=dataset_path)
        assert len(source_text_list) == len(target_text_list), "Err: Number of source and target is different"

        # create tag dictionary
        tag_to_index = self._create_tag_dict(total_target_text_list=target_text_list,
                                             caching_tag_dict_path=caching_tag_dict_path)

        self.source_text_list = source_text_list
        self.target_text_list = target_text_list
        self.tag_to_index = tag_to_index

    def __len__(self):
        return len(self.source_text_list)

    def __getitem__(self, data_idx: int):
        # TODO
        # input_str, input_token_indexes, input_tag_indexes
        return None

    def _load_dataset_from_files(self, dataset_path: str) -> (list, list):
        txt_filenames = os.listdir(dataset_path)
        txt_filenames = list(filter(lambda filename: filename.endswith(".txt"), txt_filenames))
        txt_filepaths = list(map(lambda filename: os.path.join(dataset_path, filename), txt_filenames))

        total_source_text_list = []
        total_target_text_list = []
        for txt_filepath in txt_filepaths:
            source_text_list, target_text_list = self._load_text_and_label_from_txt(txt_filepath)
            total_source_text_list += source_text_list
            total_target_text_list += target_text_list

        return total_source_text_list, total_target_text_list

    def _load_text_and_label_from_txt(self, txt_filepath: str) -> (list, list):
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

        return source_text_list, target_text_list

    def _create_tag_dict(self, total_target_text_list: [str], caching_tag_dict_path: str = None):
        if caching_tag_dict_path is not None and os.path.exists(caching_tag_dict_path):
            try:
                f = open(caching_tag_dict_path, 'r')
                return json.load(f)
            except:
                print(f"Err: fail to load tag dictionary from '{caching_tag_dict_path}'")
                pass

        tag_to_index = self._default_tag_to_index
        toal_tags = []
        for total_target_text in total_target_text_list:
            tags = re.findall(r'\<(.*?)\>', total_target_text)  # find all content between < and >
            tags = list(filter(lambda tag: len(tag.split(":")) == 2, tags))
            tags = list(map(lambda tag: tag.split(":")[-1], tags))  # only tag names
            tags = list(dict.fromkeys(tags))  # remove duplications
            for tag in tags:
                if tag not in toal_tags:
                    toal_tags.append(tag)
        for tag in toal_tags:
            tag_to_index[f'B-{tag}'] = len(tag_to_index)
            tag_to_index[f'I-{tag}'] = len(tag_to_index)

        if caching_tag_dict_path is not None:
            f = open(caching_tag_dict_path, 'w')
            json.dump(tag_to_index, f, ensure_ascii=False, indent=4)
        return tag_to_index


if __name__ == '__main__':
    dataset_path = "data_in/NER-master/말뭉치 - 형태소_개체명"
    print(os.listdir(dataset_path))
    dataset = NERDataset(dataset_path=dataset_path)
