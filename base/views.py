from django.http import response
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Predictor,SentimentAnalyzer,FakeNewsDetector,Summarizer,Chatbot
import json
import pandas as pd
from django.http import JsonResponse
# Create your views here.

def home_view(request):
    return render(request,'home.html')

@csrf_exempt
def predict_view(request):
    if request.method=='POST':
        predictor = Predictor()
        dataset = predictor.load_data()
        test_result,forecast_result = predictor.predict(dataset)
        accuracy = predictor.get_accuracy(test_result)
        test_plot_div = predictor.visualize_test(test_result)
        forecast_plot_div = predictor.visualize_forecast(dataset,forecast_result)
        forecast_result.Date_reported=forecast_result.Date_reported.dt.strftime('%Y-%m-%d')
        forecasted_json = forecast_result.reset_index().to_json(orient ='records') 
        forecast_result = [] 
        forecast_result = json.loads(forecasted_json) 
        ctx = {'accuracy':accuracy,'dataset':dataset,'forecast_result':forecast_result,'test_div':test_plot_div,'forecast_div':forecast_plot_div}
        return render(request,'predict.html',context=ctx)
    return render(request,'predict.html')

@csrf_exempt
def sentiment_view(request):
    if request.method=='POST':
        analyzer = SentimentAnalyzer()
        tweets = analyzer.get_tweets()
        tweets = analyzer.clean_tweets(tweets)
        result = analyzer.get_sentiment(tweets)
        freq_plot_div = analyzer.visualize(result)
        result_json = result.reset_index().to_json(orient ='records') 
        result = [] 
        result = json.loads(result_json)
        ctx = {'result':result,'freq_plot_div':freq_plot_div}
        return render(request,'sentiment.html',context=ctx)
    return render(request,'sentiment.html')

@csrf_exempt
def summary_view(request):
    if request.method=='POST':
        text = request.POST.get('article')
        summarizer = Summarizer()
        summary = summarizer.summarize(text)
        ctx = {'article' :text, 'summary':summary,'article_count':len(text),'summary_count':len(summary)}
        return render(request,'summary.html',ctx)
    return render(request,'summary.html')

@csrf_exempt
def fake_news_view(request):
    if request.method=='POST':
        text = request.POST.get('news')
        detector = FakeNewsDetector()
        dataset = detector.load_data()
        matrix,accuracy,result = detector.detect(dataset,text)
        ctx = {'matrix':matrix,'accuracy':accuracy,'result':result}
        return render(request,'fakenews.html',ctx)
    return render(request,'fakenews.html')

chatbot = Chatbot()
dataset = chatbot.load_data()
model,response_encodings = chatbot.load_model(dataset)

def chatbot_view(request):
    if request.is_ajax():
        question = request.GET.get('msg')
        global model
        global response_encodings
        global chatbot
        global dataset
        bot_response = chatbot.get_response(model,response_encodings,question,dataset)
        ans = {'bot_response': bot_response}
        return JsonResponse(ans)
    return render(request,'chatbot.html')
