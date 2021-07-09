from django.shortcuts import render

# Create your views here.

def home_view(request):
    return render(request,'home.html')

def predict_view(request):
    return render(request,'predict.html')

def sentiment_view(request):
    return render(request,'sentiment.html')

def summary_view(request):
    return render(request,'summary.html')

def fake_news_view(request):
    return render(request,'fakenews.html')

def chatbot_view(request):
    return render(request,'chatbot.html')