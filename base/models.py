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