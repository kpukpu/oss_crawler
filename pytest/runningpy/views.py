from multiprocessing import context
from django.shortcuts import render, HttpResponse
import requests
from bs4 import BeautifulSoup
from summarizer import Summarizer

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
    model = Summarizer()
    summar = model(newscontext)
    return HttpResponse(summar)