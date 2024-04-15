from django.urls import path
from runningpy import views

urlpatterns = [
    path('',views.index),
    path('calculate/',views.calculate),
    path('result/',views.result),
]
