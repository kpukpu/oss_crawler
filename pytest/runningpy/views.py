from multiprocessing import context
from django.shortcuts import render, HttpResponse
import requests
from bs4 import BeautifulSoup
from summarizer import Summarizer
import nltk
from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
from PIL import Image
import numpy as np
import sys
import jpype
import io
import matplotlib.pyplot as plt
# Create your views here.

newscontext = " "

def index(request):
    return render(request, 'runningpy/base.html')


def calculate(request):
    try:
        global newscontext
        if request.method == 'POST':
            n_link= request.POST.get('n1')
        # HTTP GET 요청을 보내고 응답을 받음
        response = requests.get(n_link) 
        response.raise_for_status()  # 오류가 있을 경우 예외 발생

        # BeautifulSoup 객체 생성, HTML 파서로 'html.parser' 사용
        soup = BeautifulSoup(response.text, 'html.parser')

        # 'id'가 'dic_area'인 <article> 태그 찾기
        content_article = soup.find('article', id='dic_area')
        # <article> 태그 내의 모든 텍스트 추출
        news_content = ' '.join(content_article.stripped_strings)
        newscontext = news_content
        return render(request, 'runningpy/crawl.html', {'news_content': news_content})
    except requests.RequestException as e:
        return f"HTTP Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

'''
def calculate(request):
    if request.method == 'POST':
        number1 = request.POST.get('n1')
        
    return HttpResponse('결과 : ' + str(result))
'''
#def summarize(request):
    
def result(request):
    return HttpResponse('''''')


def summarize(request):
    model_name = "google/bert_uncased_L-4_H-256_A-4"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

# 문장별로 처리
    sentences = sent_tokenize(newscontext)

    if len(sentences) < 3:
        return HttpResponse("Not enough sentences to summarize.")
    
    sentence_embeddings = []
    for sentence in sentences:
        encoded_input = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            output = model(**encoded_input)
        sentence_embedding = torch.mean(output.last_hidden_state, dim=1)
        sentence_embeddings.append(sentence_embedding)

# 문장 중요도 계산
    sentence_scores = torch.cat(sentence_embeddings, dim=0)
    k = min(3, len(sentences)) 
    important_sentence_indices = torch.topk(torch.norm(sentence_scores, dim=1), k).indices

# 중요한 문장 출력
    result = " "
    important_sentences = [sentences[index] for index in important_sentence_indices]
    for sentence in important_sentences:
        result += sentence
    return HttpResponse(result)




def _wordcloud(request):
    least_num = 2#3번 이상 호출된 단어만 워드 클라우드에 출력
    
#    temp_save_dirc = 'C:\\Users\\kpukpu\\Desktop'

#저장 주소 처리
#    save_empty_list = []
#    save_empty_str = ""
#    for i in temp_save_dirc:
#        if(i == "\\"):
#            i = '/'
#            save_empty_list.append(i)
#        else:
#            save_empty_list.append(i)
#   real_save_dirc = save_empty_str.join(save_empty_list)
#    real_save_dirc = real_save_dirc + "/Word_cloud.png"

#matplotlib 대화형 모드 켜기
    plt.ion()

    text = newscontext
# OKT 사전 설정
    okt = Okt()

#명사만 추출
    nouns = okt.nouns(text)

# 단어의 길이가 1개인 것은 제외
    words = [n for n in nouns if len(n) > 1]

# 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구함
    c = Counter(words)


#최소 빈도수 처리
    key = list(c.keys())
    for a in key:
        if(c[a] < least_num):
            del c[a]

#빈도수가 맞지 않을 시 프로그램을 종료
    if(len(c) == 0):
        print("최소 빈도수가 너무 큽니다. 다시 설정해 주세요.")
        print("프로그램을 종료합니다.")
        sys.exit()

#워드클라우드 만들기
    wc = WordCloud(background_color="white" ,  font_path=r"C:/Windows/Fonts/malgun.ttf", width=600, height=600, scale=2.0, max_font_size=250)
    gen = wc.generate_from_frequencies(c)
#    plt.figure()
#    plt.imshow(gen)
    buffer = io.BytesIO()
    gen.to_image().save(buffer, format='PNG')
    buffer.seek(0)
#파일로 저장 
    
    return HttpResponse(buffer.getvalue(), content_type='image/png')