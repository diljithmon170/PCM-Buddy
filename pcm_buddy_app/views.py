from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from .pcm_chatbot.utils import get_answer  # Import your chatbot logic

def home(request):
    return render(request, 'home.html')

def chat_physics(request):
    return render(request, 'chatbot.html', {'subject': 'Physics'})

def chat_chemistry(request):
    return render(request, 'chatbot.html', {'subject': 'Chemistry'})

def chat_maths(request):
    return render(request, 'chatbot.html', {'subject': 'Maths'})

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        data = json.loads(request.body)
        user_message = data.get("message", "")
        subject = data.get("subject", "")
        marks = int(data.get("marks", 2))  # Default to 2 if not provided
        bot_reply = get_answer(user_message, subject, marks)
        return JsonResponse({"reply": bot_reply})
    return JsonResponse({"reply": "Invalid request."}, status=400)
