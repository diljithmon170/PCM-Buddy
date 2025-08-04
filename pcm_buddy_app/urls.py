from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/physics/', views.chat_physics, name='chat_physics'),
    path('chat/chemistry/', views.chat_chemistry, name='chat_chemistry'),
    path('chat/maths/', views.chat_maths, name='chat_maths'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('login/', views.log_sig, name='log_sig'),
    path('logout/', views.logout_view, name='logout'),
]