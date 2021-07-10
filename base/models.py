from django.db import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.contrib.staticfiles import finders
from covision.settings import BASE_DIR
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import datetime
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import neattext.functions as nf
from textblob import TextBlob
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle
# Create your models here.

class Predictor():
    def load_data(self):
        csv_path = os.path.join(BASE_DIR,'base','datasets','covid_cases.csv')
        dataset = pd.read_csv(csv_path)
        dataset = dataset[['Date_reported','New_cases']]
        return dataset

    def predict(self,dataset):
        X_dataset_train = dataset[0:int(len(dataset)*0.75)]
        X_dataset_test = dataset[int(len(dataset)*0.75):]
        training=pd.DataFrame(X_dataset_train['New_cases'])
        testing=pd.DataFrame(X_dataset_test['New_cases'])
        sc = StandardScaler()
        training = sc.fit_transform(training)
        testing = sc.transform(testing)
        window=30
        x_train=[]
        y_train=[]
        for i in range(window,len(training)):
            x_train.append(training[i-window:i,0])  
            y_train.append(training[i,0])
        x_train,y_train=np.array(x_train),np.array(y_train)
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        x_test=[]
        for i in range(window,len(testing)):
            x_test.append(testing[i-window:i,0])  
        x_test=np.array(x_test)
        y_pred = sc.inverse_transform(regressor.predict(x_test))
        Final = X_dataset_test[window:]
        Final['Predicted'] = y_pred
        X_dataset_test['Date_reported']=pd.to_datetime(X_dataset_test['Date_reported'],format="%Y-%m-%d")
        for i in range (window):
            x_test=[]
            x_test.append(testing[-window:,0])  
            x_test=np.array(x_test)
            pred=sc.inverse_transform(regressor.predict(x_test))
            new_date=X_dataset_test['Date_reported'][-1:] + datetime.timedelta(days=1)
            data = {'Date_reported':new_date.values[0],'New_cases':pred[0]}
            X_dataset_test = X_dataset_test.append(data,ignore_index=True)
            testing = np.reshape(np.append(testing,regressor.predict(x_test)),(-1,1))
        X_dataset_test['New_cases'][-window:] = X_dataset_test['New_cases'][-window:].apply(lambda x: int(round(x)))
        return Final,X_dataset_test[-window:]

    def get_accuracy(self,result):
        return round(r2_score(result['New_cases'], result['Predicted'])*100,2)
    
    def visualize_test(self,Final):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Final['Date_reported'],y=Final['New_cases'],name='Actual'))
        fig.add_trace(go.Scatter(x=Final['Date_reported'],y=Final['Predicted'],name='Predicted'))
        fig.update_layout(title='Daily Cases VS Date',xaxis_title='Date',yaxis_title='Number of Daily Cases')
        plot_div=plot(fig, output_type='div',include_plotlyjs=False)
        return plot_div

    def visualize_forecast(self,dataset,result):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dataset['Date_reported'],y=dataset['New_cases'],name='Prehistoric Data'))
        fig.add_trace(go.Scatter(x=result['Date_reported'],y=result['New_cases'],name='Forecasted Data'))
        fig.update_layout(title='Daily Cases VS Date',xaxis_title='Date',yaxis_title='Number of Daily Cases')
        plot_div=plot(fig, output_type='div',include_plotlyjs=False)
        return plot_div

tweet_dataset = pd.DataFrame()

class Listener(StreamListener):
    def __init__(self,filename):
        super(Listener,self).__init__()
        self.counter = 0
        self.filename = filename

    def on_status(self, status):
        global tweet_dataset
        if self.counter < 10:
            try:
                if hasattr(status, 'retweeted_status') and hasattr(status.retweeted_status, 'extended_tweet'):
                    tweet_dataset = tweet_dataset.append({'tweet':status.retweeted_status.extended_tweet['full_text']},ignore_index=True)
                    self.counter+=1
                if hasattr(status, 'extended_tweet'):
                    tweet_dataset = tweet_dataset.append({'tweet':status.extended_tweet['full_text']},ignore_index=True)
                    self.counter+=1
                else:
                    pass
            except AttributeError:
                pass
        else:
            return False

    def on_error(self,status):
        if status==420:
            print(status)
            return False
        
class TwitterStreamer():
    API_KEY = '46bd35ovqNJBO5ny5xSu0IY97'
    API_SECRET_KEY = 'm62qpyHEcFRagq1H37SdkwH8IQFZgCXYjXmKAFku2NdcIXF4vz'
    ACCESS_TOKEN = '1413146033991733265-LqS7YwKXgRQoeDYQyNxgE0atJ52onx'
    ACCESS_TOKEN_SECRET = 'CU1e7KlTIfbzkKtrCjzsmcwh1LqyclNPH9AVq5XFZtCY7'

    def stream_tweets(self,filename,hashtags):
        listener = Listener(filename)
        auth = OAuthHandler(self.API_KEY,self.API_SECRET_KEY)
        auth.set_access_token(self.ACCESS_TOKEN,self.ACCESS_TOKEN_SECRET)
        stream = Stream(auth,listener,tweet_mode='extended')
        stream.filter(track=hashtags,languages=['en'])


class SentimentAnalyzer():
    def get_tweets(self):
        hashtags = ['covid','covid19','COVID19','CORONAVIRUS','coronavirus','COVID']
        filename = 'tweets.json'
        twitter_streamer = TwitterStreamer()
        twitter_streamer.stream_tweets(filename,hashtags)
        global tweet_dataset
        return tweet_dataset

    def clean_tweets(self,dataset):
        dataset['tweet'] = dataset['tweet'].apply(nf.remove_hashtags)
        dataset['tweet'] = dataset['tweet'].apply(lambda x: nf.remove_userhandles(x))
        dataset['tweet'] = dataset['tweet'].apply(nf.remove_multiple_spaces)
        dataset['tweet'] = dataset['tweet'].apply(nf.remove_urls)
        dataset['tweet'] = dataset['tweet'].apply(nf.remove_puncts)
        dataset['tweet'] = dataset['tweet'].apply(nf.remove_emojis)
        dataset['tweet'] = dataset['tweet'].apply(lambda x: x.replace('RT ',''))
        return dataset

    def get_sentiment(self,dataset):
        sentiment = []
        polarity = []
        for tweet in dataset['tweet']:
            analysis = TextBlob(tweet)
            if analysis.sentiment.polarity>0:
                senti = 'Positive'
            elif analysis.sentiment.polarity==0:
                senti = 'Neutral'
            else:
                senti = 'Negative'
            sentiment = np.append(sentiment,senti)
            polarity = np.append(polarity,analysis.sentiment.polarity)
        dataset['polarity'] = polarity
        dataset['sentiment'] = sentiment 
        return dataset
    
    def visualize(self,dataset):
        neg = dataset[dataset['sentiment']=='Negative']['sentiment'].count()
        pos = dataset[dataset['sentiment']=='Positive']['sentiment'].count()
        neu = dataset[dataset['sentiment']=='Neutral']['sentiment'].count()
        freq = {'Positive':pos,'Negative':neg,'Neutral':neu}
        X = list(freq.keys())
        Y = list(freq.values())
        fig = px.bar(x=X, y=Y)
        fig.update_layout(title='Count of Positive/Negative/Neutral Tweets Related To Covid-19',xaxis_title='Predicted Sentiment',yaxis_title='Count')
        plot_div=plot(fig, output_type='div',include_plotlyjs=False)
        return plot_div

class FakeNewsDetector():
    def load_data(self):
        xl_path = os.path.join(BASE_DIR,'base','datasets','fake_news.xlsx')
        dataset = pd.read_excel(xl_path)
        return dataset
    
    def detect(self,dataset,text):
        X = dataset['text']
        y = dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        tf = TfidfVectorizer(stop_words='english',max_df=0.7)
        X_train = tf.fit_transform(X_train)
        X_test = tf.transform(X_test)
        model_path = os.path.join(BASE_DIR,'base','trained_models','fakenews.pkl')
        file = open(model_path, 'rb')     
        classifier = pickle.load(file)
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = round(accuracy_score(y_test, y_pred)*100,2)
        result = 'True' if classifier.predict(tf.transform([text]))==[1] else 'False'
        return cm,accuracy,result
        return 0,0,0

        
        


