import os, json, re

from torch.utils.data import Dataset
import torch
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer

from data_utils.pad_sequence import keras_pad_fn
from data_utils.utils import download as _download
from data_utils.vocab_tokenizer import Vocabulary, Tokenizer

kobert_models = {
    'pytorch_kobert': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
        'fname': 'pytorch_kobert_2439f391a6.params',
        'chksum': '2439f391a6'
    },
    'vocab': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/vocab/kobertvocab_f38b8a4d6d.json',
        'fname': 'kobertvocab_f38b8a4d6d.json',
        'chksum': 'f38b8a4d6d'
    },
    'onnx_kobert': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/onnx/onnx_kobert_44529811f0.onnx',
        'fname': 'onnx_kobert_44529811f0.onnx',
        'chksum': '44529811f0'
    },
    'tokenizer': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/tokenizer/tokenizer_78b3253a26.model',
        'fname': 'tokenizer_78b3253a26.model',
        'chksum': '78b3253a26'
    }
}

class NERDataset(Dataset):

    _default_tag_to_index = {"[CLS]": 0, "[SEP]": 1, "[PAD]": 2, "[MASK]": 3, "O": 4}

    def __init__(self, dataset_path: str, caching_tag_dict_path: str=None, model_maxlen: int=-1):
        self.dataset_path = dataset_path

        # load dataset from file system
        source_text_list, target_text_list = self._load_dataset_from_files(dataset_path=dataset_path)
        assert len(source_text_list) == len(target_text_list), "Err: Number of source and target is different"

        vocab_of_gluonnlp = self._download_kobert_vocab()
        token_to_idx = vocab_of_gluonnlp.token_to_idx
        vocab = Vocabulary(token_to_idx=token_to_idx)
        kobert_tokernizer_path = self._download_kobert_tokenizer()  # ./tokenizer_78b3253a26.model
        ptr_tokenizer = SentencepieceTokenizer(kobert_tokernizer_path)
        tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_maxlen)

        # create tag dictionary
        tag_to_index = self._create_tag_dict(total_target_text_list=target_text_list,
                                             caching_tag_dict_path=caching_tag_dict_path)
        """
        source_text_list[0]: 오에 겐자부로는 일본 현대문학의 초석을 놓은 것으로 평가받는 작가 나쓰메 소세키(1867~1916)의 대표작 ‘마음’에 담긴 군국주의적 요소, 야스쿠니 신사 참배 행위까지 소설의 삽화로 동원하며 일본 사회의 ‘비정상성’을 문제 삼는다.
        target_text_list[0]: <오에 겐자부로:PER>는 <일본:LOC> 현대문학의 초석을 놓은 것으로 평가받는 작가 <나쓰메 소세키:PER>(<1867~1916:DUR>)의 대표작 ‘<마음:POH>’에 담긴 군국주의적 요소, <야스쿠니 신사:ORG> 참배 행위까지 소설의 삽화로 동원하며 <일본:ORG> 사회의 ‘비정상성’을 문제 삼는다.
        """
        self.source_text_list = source_text_list
        self.target_text_list = target_text_list
        self.token_to_idx = token_to_idx
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.tag_to_index = tag_to_index


    def __len__(self):
        return len(self.source_text_list)

    def __getitem__(self, data_idx: int):
        source_text = self.source_text_list[data_idx]
        target_text = self.target_text_list[data_idx]

        # ============================================
        # ============================================
        # source_text
        tokens = self.tokenizer.split(source_text)  # kobert의 SentencepieceTokenizer
        token_ids_with_cls_sep = self.tokenizer.list_of_string_to_arr_of_cls_sep_pad_token_ids([source_text])

        print()
        print("source_text:", source_text)
        # source_text: 이어 옆으로 움직여 김일성의 오른쪽에서 한 차례씩 두 번 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.
        print("tokens:", tokens)
        # tokens: ['▁이어', '▁옆', '으로', '▁', '움', '직', '여', '▁김', '일', '성', '의', '▁', '오른쪽', '에서', '▁한', '▁차례', '씩', '▁두', '▁번', '▁상', '체', '를', '▁', '굽', '혀', '▁조', '문', '했으며', '▁이', '윽', '고', '▁안', '경', '을', '▁벗', '고', '▁손', '수', '건으로', '▁눈', '주', '위를', '▁', '닦', '기도', '▁했다', '.']
        print("token_ids_with_cls_sep:", token_ids_with_cls_sep)
        """
        token_ids_with_cls_sep: [[   2 3716 3395 7078  517 7014 7342 6916 1316 7126 6573 7095  517 6967
        6903 4955 4407 6792 1773 2307 2658 7436 6116  517 5519 7899 4162 6234
        7877 3647]]
        """

        current_idx = 0
        starting_indexes_of_each_token = []
        for idx, token in enumerate(tokens):
            starting_indexes_of_each_token.append(current_idx)
            current_idx += len(token) if idx != 0 else len(token) - 1
        print("starting_indexes_of_each_token:", starting_indexes_of_each_token)
        # starting_indexes_of_each_token: [0, 2, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 19, 21, 23, 26, 27, 29, 31, 33, 34, 35, 36, 37, 38, 40, 41, 44, 46, 47, 48, 50, 51, 52, 54, 55, 57, 58, 61, 63, 64, 66, 67, 68, 70, 73]
        print()

        # ============================================
        # ============================================
        # target_text
        print("target_text:", target_text)
        # TODO
        print()

        # 이어 옆으로 움직여 김일성의 오른쪽에서 한 차례씩 두 번 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.
        # 이어 옆으로 움직여 <김일성:PER>의 오른쪽에서 <한 차례:NOH>씩 <두 번:NOH> 상체를 굽혀 조문했으며 이윽고 안경을 벗고 손수건으로 눈주위를 닦기도 했다.


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

    def _download_kobert_vocab(self, cachedir='./ptr_lm_model'):
        # download vocab
        vocab_info = kobert_models['vocab']
        vocab_path = _download(vocab_info['url'],
                               vocab_info['fname'],
                               vocab_info['chksum'],
                               cachedir=cachedir)
        vocab_b_obj = nlp.vocab.BERTVocab.from_json(open(vocab_path, 'rt').read())
        return vocab_b_obj

    def _download_kobert_tokenizer(self, cachedir='./ptr_lm_model'):
        """Get KoBERT Tokenizer file path after downloading
        """
        tokenizer_info = kobert_models['tokenizer']
        return _download(tokenizer_info['url'],
                         tokenizer_info['fname'],
                         tokenizer_info['chksum'],
                         cachedir=cachedir)

if __name__ == '__main__':
    dataset_path = "data_in/NER-master/말뭉치 - 형태소_개체명"
    print(os.listdir(dataset_path))
    dataset = NERDataset(dataset_path=dataset_path, model_maxlen=30)
    print(dataset.__getitem__(0))