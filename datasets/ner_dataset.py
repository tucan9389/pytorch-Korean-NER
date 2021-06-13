import os
import torch

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