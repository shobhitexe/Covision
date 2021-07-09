from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Predictor
import json
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

def sentiment_view(request):
    return render(request,'sentiment.html')

def summary_view(request):
    return render(request,'summary.html')

def fake_news_view(request):
    return render(request,'fakenews.html')

def chatbot_view(request):
    return render(request,'chatbot.html')