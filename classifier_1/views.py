from django.shortcuts import render
import re # 사용자 텍스트 전처리
from keras.models import load_model # 모델 불러오기
from tensorflow import keras # 예측용
import numpy as np # 예측용
from tensorflow.keras.preprocessing.sequence import pad_sequences # 사용자 텍스트 패딩
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json # 사용자 텍스트 토큰화
import json # keras의 tokenizer 불러오기
from kiwipiepy import Kiwi # 사용자 텍스트 형태소 분석
from kiwipiepy.utils import Stopwords # 불용어 처리용


# Create your views here.
def classifier_1(request):
    if request.GET.get("text", None) == None:
        return render(request, 'classifier.html')

    else:
        dft_text = request.GET.get("text")
        text = dft_text[:]
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

    stop_words = []
    with open("classifier_1/final_stop_words.txt", "r", encoding = "utf-8") as f:    
        for word in f:
            stop_words.append(word.strip())
    stop_words = list(set(stop_words)) # 중복 제거

    # 불용어 객체 생성 및 단어 추가
    sws = Stopwords()
    for word in stop_words:
        sws.add(word)
    # 형태소 분석기 가져오기
    kiwi = Kiwi()


    # 커스텀 단어사전 적용
    kiwi.add_user_word("classifier_1/komoran_base_user_dict.txt")

    # 사용자 입력 데이터 형태소 분석
    kiwi_news = kiwi.tokenize(text, normalize_coda = True, stopwords = sws)

    # 불용어 제거 및 명, 형, 동만 남기기
    final_text = []
    for word_set in kiwi_news:
        if len(word_set) > 0:
            word = word_set[0]
            tag = word_set[1]
            if tag in ("NNP", "NNG", "VV", "VA"):
                final_text.append(word)

    # 문장으로 만들기
    final_seq = []
    temp_seq = ""
    for seq in final_text:
        temp_seq += seq + " "
    temp_seq = temp_seq[:-1]
    final_seq.append(temp_seq)

    # 토큰화
    with open("classifier_1/tokenizer_kiwi.json", "r", encoding = "utf-8") as f:
        token_dict = json.load(f)
        tokenizer = tokenizer_from_json(token_dict)
    user_text = tokenizer.texts_to_sequences(final_seq)

    # 패딩
    user_text = pad_sequences(user_text, maxlen = 300, truncating = "post", padding = "post")

    # 모델 불러오기
    model = load_model("classifier_1/model/cnn_model.keras")

    # 예측 진행
    pred = model.predict(user_text)
    pred_result = np.argmax(pred)

    if pred_result == 0:
        bosu="보수 성향의 기사입니다."
        context = {
            "text" : dft_text,
            "bosu" : bosu

        }

    else:
        jinbo="진보 성향의 기사입니다."
            
        context = {
            "text" : dft_text,
            "jinbo" : jinbo
        }

    return render(request, 'classifier.html', context)