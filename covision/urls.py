"""covision URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls.conf import include
from base.views import home_view,predict_view,sentiment_view,summary_view,fake_news_view,chatbot_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('predict', predict_view, name='predict'),
    path('sentiment', sentiment_view, name='sentiment'),
    path('summary', summary_view, name='summary'),
    path('fake_news', fake_news_view, name='fakenews'),
    path('chatbot', chatbot_view, name='chatbot'),
]
