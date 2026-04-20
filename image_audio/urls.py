from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/encode/', views.encode_image, name='encode_image'),
    path('api/decode/', views.decode_audio, name='decode_audio'),
    path('api/decode/stream/', views.decode_audio_stream, name='decode_audio_stream'),
]