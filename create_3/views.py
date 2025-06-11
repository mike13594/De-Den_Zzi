from django.shortcuts import render
from switch_sides_2.forms import OnOffForm
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
import torch


tokenizer_b = PreTrainedTokenizerFast.from_pretrained("create_3/bosu/",
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>')

model_b = AutoModelWithLMHead.from_pretrained("create_3/bosu/")

tokenizer_j = PreTrainedTokenizerFast.from_pretrained("create_3/jinbo/",
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>')

model_j = AutoModelWithLMHead.from_pretrained("create_3/jinbo/")

# Create your views here.
def create_3(request):
    onoffform = OnOffForm(request.GET or None)

    if request.GET.get("on_off", None) == None:
        context = {
        "onoffform" : onoffform,
    }


    elif request.GET.get("on_off", "") == '진보':
        text = request.GET.get("text")
        input_ids = tokenizer_j.encode(text)
        gen_ids = model_j.generate(torch.tensor([input_ids]),
                           max_length=int(request.GET.get("length")),
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer_j.pad_token_id,
                           eos_token_id=tokenizer_j.eos_token_id,
                           bos_token_id=tokenizer_j.bos_token_id,
                           use_cache=True
                        )
        generated = ".".join(tokenizer_j.decode(gen_ids[0,:].tolist()).split(".")[:-1])+"."

        context = {
            "onoffform" : onoffform,
            "generated_j" : generated,
            "text" : text,
        }

    elif request.GET.get("on_off", "") == '보수':
        text = request.GET.get("text")
        input_ids = tokenizer_b.encode(text)
        gen_ids = model_b.generate(torch.tensor([input_ids]),
                           max_length=int(request.GET.get("length")),
                           repetition_penalty=2.0,
                           pad_token_id=tokenizer_b.pad_token_id,
                           eos_token_id=tokenizer_b.eos_token_id,
                           bos_token_id=tokenizer_b.bos_token_id,
                           use_cache=True
                        )
        generated = ".".join(tokenizer_b.decode(gen_ids[0,:].tolist()).split(".")[:-1])+"."

        context = {
            "onoffform" : onoffform,
            "generated_b" : generated,
            "text" : text,
        }
    return render(request, 'create.html', context)

