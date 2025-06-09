from django.apps import AppConfig
import html
import pathlib
import os
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast


class Create3Config(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "create_3"

# class kogpt_j(AppConfig):
#     name = 'kogpt_j'
#     tokenizer = PreTrainedTokenizerFast.from_pretrained("./jinbo",
#                                                         bos_token='</s>', eos_token='</s>', unk_token='<unk>',
#                                                         pad_token='<pad>', mask_token='<mask>')
#     model = AutoModelWithLMHead.from_pretrained("./jinbo")