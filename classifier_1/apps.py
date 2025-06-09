from django.apps import AppConfig
import html
import pathlib
import os
import transformers

# from fast_bert.prediction import BertClassificationPredictor

class Classifier1Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier_1"

# class kogpt_j(AppConfig):
#     name = 'kogpt_j'
#     MODEL_PATH = Path("model")
#     KOBERT_PRETRAINED_PATH = Path("./model_jinbo")
#     LABEL_PATH = Path("label/")
#     predictor = BertClassificationPredictor(model_path = MODEL_PATH/"multilabel-emotion-fastbert-basic.bin", 
#                                             pretrained_path = BERT_PRETRAINED_PATH, 
#                                             label_path = LABEL_PATH, multi_label=True)  