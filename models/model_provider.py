from transformers import (
    BertConfig, BertForTokenClassification, BertTokenizer, # for KcBERT
    ElectraConfig, ElectraTokenizer, ElectraForTokenClassification,
    AutoConfig, AutoTokenizer, AutoModelForTokenClassification, 
)

def get_model(model_name="beomi/kcbert-base"):
    ModelConfig = None
    Tokenizer = None
    Model = None
    if model_name == "beomi/kcbert-base":
        ModelConfig = BertConfig
        Tokenizer   = BertTokenizer
        Model       = BertForTokenClassification
    elif model_name == "monologg/koelectra-base-discriminator" or model_name == "monologg/koelectra-base-v3-discriminator":
        ModelConfig = ElectraConfig
        Tokenizer   = ElectraTokenizer
        Model       = ElectraForTokenClassification
    elif model_name == "monologg/kobert":
        ModelConfig = AutoConfig
        Tokenizer   = AutoTokenizer
        Model       = AutoModelForTokenClassification
    return ModelConfig, Tokenizer, Model