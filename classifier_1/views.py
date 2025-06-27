from django.shortcuts import render
import re # 사용자 텍스트 전처리
import joblib # 모델 불러오기 용도
from kiwipiepy import Kiwi # 사용자 텍스트 형태소 분석
from kiwipiepy.utils import Stopwords # 불용어 처리용

# 불용어 리스트 생성
stop_words = []
with open("classifier_1/final_stop_words.txt", "r", encoding = "utf-8") as f:    
    for word in f:
        stop_words.append(word.strip())
stop_words = list(set(stop_words)) # 중복 제거

# kiwi 객체 생성
kiwi = Kiwi()

# 단어 추가
with open("classifier_1/komoran_base_user_dict.txt", "r", encoding = "utf-8") as f:
    for word in f:
        kiwi.add_user_word(word.strip())

# 불용어 객체 생성
sws = Stopwords()
for word in stop_words:
    sws.add(word)

# tag 필터링 딕셔너리 생성
valid_tags = {"NNP", "NNG", "NP","VV", "VA"}

# 토큰화 함수
def kiwi_token(text):    
    tokens = kiwi.tokenize(text, normalize_coda = True, stopwords = sws)
    return [token.form for token in tokens
            if (token.tag in valid_tags) and (len(token.form) > 1)]

# Create your views here.
def classifier_1(request):
    if request.GET.get("text", None) == None:
        return render(request, 'classifier.html')

    else:
        dft_text = request.GET.get("text")
        text = dft_text[:]
        text = text.replace("\n", " ")
        text = re.sub(r"\s*[가-힣]{2,4}\s*기자\s*", "", text)
        text = re.sub(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
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
        text = re.sub(r"[^가-힣一-龥A-Za-z0-9\s\(\)\-\:\[\]\'\"\.\,\·\~]+", "", text)
        remove_t1 = "\[[^]]*\]" # 대괄호
        remove_t2 = "\([^)]*\)" # 소괄호
        text = re.sub(remove_t1, "", text)
        text = re.sub(remove_t2, "", text)
        user_text = text.replace("  ", " ")


    # 사용자 입력 데이터 토큰화 + 명, 형, 동만 남기기
    final_seq = " ".join(kiwi_token(user_text))

    # tfidvectorizer 불러오기 및 사용자 입력 데이터 할당
    tfid_vect = joblib.load("classifier_1/tfid_vectorizer.joblib")
    user_text = tfid_vect.transform([final_seq])

    # 모델 불러오기
    model = joblib.load("classifier_1/model/svc_classifier.joblib")

    # 예측 진행
    pred = model.predict(user_text)

    # 결과 출력
    if pred[0] == 0:
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