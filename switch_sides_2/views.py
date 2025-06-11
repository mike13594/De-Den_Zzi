from django.shortcuts import render
from switch_sides_2.forms import OnOffForm
import re
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, AutoModelWithLMHead

# 보수 모델, 토크나이저
tokenizer_b = PreTrainedTokenizerFast.from_pretrained("create_3/bosu/",
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>')

model_b = AutoModelWithLMHead.from_pretrained("create_3/bosu/")

# 진보 모델, 토크나이저
tokenizer_j = PreTrainedTokenizerFast.from_pretrained("create_3/jinbo/",
bos_token='</s>', eos_token='</s>', unk_token='<unk>',
pad_token='<pad>', mask_token='<mask>')

model_j = AutoModelWithLMHead.from_pretrained("create_3/jinbo/")

# 사용자 텍스트 전처리 함수
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s*[가-힣]{2,4}\s*기자\s*", "", string = text)
    text = re.sub(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", string = text)
    text = text.replace("...", ".")
    text = text.replace(".,", ".")
    text = text.replace(".  ", ".")
    text = text.replace(". ", ".")
    text = text.replace(" .", ".")
    text = text.replace(".", ". ")
    text = text.replace(",  ", ",")
    text = text.replace(", ", ",")
    text = text.replace(",", ", ")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace(" · ", "·")
    text = text.replace("·", " · ")
    text = text.replace(" / ", "/")
    text = text.replace("/", " / ")
    text = re.sub(r"[^가-힣一-龥A-Za-z0-9\s\(\)\-\:\[\]\'\"\.\,\·\~]+", "", string = text)
    remove_t1 = "\[[^]]*\]" # 대괄호
    remove_t2 = "\([^)]*\)" # 소괄호
    text = re.sub(remove_t1, "", string = text)
    text = re.sub(remove_t2, "", string = text)
    text = text.replace("  ", " ")

    return text

class KoBARTConditionalGeneration(torch.nn.Module):
    def __init__(self):
        super(KoBARTConditionalGeneration, self).__init__()
        
        # KoBART 모델 불러오기
        self.model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
        # KoBART 사전훈련 토크나이저 불러오기
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
        # 패딩 토큰 ID 가져오기 → 입력 데이터에서 패딩된 부분 식별용
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def forward(self, inputs):
        # attention_mask : 패딩이 아닌 부분운 1, 패딩인 부분은 0으로 마스킹
        attention_mask = inputs["input_ids"].ne(self.pad_token_id).float()
        # 디코더 입력에 대해서도 동일하게 attention mask 생성
        decoder_attention_mask = inputs["decoder_input_ids"].ne(self.pad_token_id).float()
        # 딕셔너리 형태로 결과 반환
        return self.model(input_ids = inputs["input_ids"],
                          attention_mask = attention_mask,
                          decoder_input_ids = inputs["decoder_input_ids"],
                          decoder_attention_mask = decoder_attention_mask,
                          labels = inputs["labels"], return_dict = True)

# Create your views here.
def switch_sides_2(request):
    onoffform = OnOffForm(request.GET or None)
    
    if request.GET.get("on_off", None) == None:
        context = {
        "onoffform" : onoffform,
    }

    elif request.GET.get("on_off", "") == '진보':
        text = request.GET.get("text")
        text = clean_text(text)
        # device 지정
        device = torch.device("cpu")

        # 모델 가중치 설정
        model = KoBARTConditionalGeneration().to(device)
        model.load_state_dict(torch.load("switch_sides_2/model/best_bart_model-14534.pt", map_location = "cpu"))

        # 모델 평가 모드로 전환
        model.eval()

        # 토크나이저 설정
        tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")

        # 텍스트 토큰화
        tokenized_text = tokenizer(text, max_length = 200, truncation=True, padding = "max_length", return_tensors = "pt")
        input_ids = tokenized_text["input_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

        output = model.model.generate(input_ids, eos_token_id = 1, max_length = 100, num_beams = 5)
        output = tokenizer.decode(output[0], skip_special_tokens = True)
        # 요약한걸 다시 생성 모델로
        input_ids = tokenizer_j.encode(output)
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
        text = clean_text(text)
        # device 지정
        device = torch.device("cpu")

        # 모델 가중치 설정
        model = KoBARTConditionalGeneration().to(device)
        model.load_state_dict(torch.load("switch_sides_2/model/best_bart_model-14534.pt", map_location = "cpu"))

        # 모델 평가 모드로 전환
        model.eval()

        # 토크나이저 설정
        tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")

        # 텍스트 토큰화
        tokenized_text = tokenizer(text, max_length = 200, truncation=True, padding = "max_length", return_tensors = "pt")
        input_ids = tokenized_text["input_ids"].to(device)
        attention_mask = tokenized_text["attention_mask"].to(device)

        output = model.model.generate(input_ids, eos_token_id = 1, max_length = 100, num_beams = 5)
        output = tokenizer.decode(output[0], skip_special_tokens = True)
        # 요약한걸 다시 생성 모델로
        input_ids = tokenizer_b.encode(output)
        gen_ids = model_b.generate(torch.tensor([input_ids]),
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
            "generated_b" : generated,
            "text" : text,
        }


    return render(request, 'switch_sides.html', context)