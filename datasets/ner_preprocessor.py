from tqdm import tqdm

class NERPreprocessor(object):
    @staticmethod
    def preprocess(tokenizer, splited_texts_list, splited_tags_list, label_map, max_seq_len, pad_token_label_id=-100):  # bert의 최대 maxlen을 사용하는것을 
        list_of_token_ids = []
        list_of_attension_mask = []
        list_of_label_ids = []

        before_text = ' '
        for splited_texts, splited_tags in tqdm(zip(splited_texts_list, splited_tags_list)):
            
            tokens_for_a_sentence = []
            tags_for_a_sentence = []
            for text, tag in zip(splited_texts, splited_tags):
                if text == ' ':
                    continue
                if before_text != ' ':
                    tokens = tokenizer.tokenize('##' + text)
                else:
                    tokens = tokenizer.tokenize(text)
                tokens_for_a_sentence += tokens
                tags_for_a_sentence += list(map(lambda x: 'O' if tag == 'O' else (f'B-{tag}' if x==0 else f'I-{tag}'), range(len(tokens))))

            
            if len(tokens_for_a_sentence) > max_seq_len - 2:
                tokens_for_a_sentence = tokens_for_a_sentence[:max_seq_len - 2]
                tags_for_a_sentence = tags_for_a_sentence[:max_seq_len - 2]
            
            tokens_for_a_sentence = [tokenizer.cls_token] + tokens_for_a_sentence + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens_for_a_sentence)
            padding_length = max_seq_len - len(token_ids)

            attension_mask = [1]*len(token_ids) + [0]*padding_length
            token_ids += [tokenizer.pad_token_id]*padding_length

            label_ids = list(map(lambda x: label_map[x], tags_for_a_sentence))
            label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
            label_ids += [pad_token_label_id]*padding_length

            list_of_token_ids.append(token_ids)
            list_of_attension_mask.append(attension_mask)
            list_of_label_ids.append(label_ids)
        
        return list_of_token_ids, \
            list_of_attension_mask, \
            list_of_label_ids