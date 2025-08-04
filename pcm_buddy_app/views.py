from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.db import models, IntegrityError

from .pcm_chatbot.utils import get_answer  # Import your chatbot logic
from .models import UserAccount

def home(request):
    user = None
    if request.session.get('user_id'):
        try:
            user = UserAccount.objects.get(id=request.session['user_id'])
        except UserAccount.DoesNotExist:
            user = None
    return render(request, "home.html", {"user": user})

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
        marks = int(data.get("marks", 2))
        force_outside = data.get("force_outside", False)
        bot_reply = get_answer(user_message, subject, marks, force_outside)
        return JsonResponse({"reply": bot_reply})
    return JsonResponse({"reply": "Invalid request."}, status=400)

def log_sig(request):
    error = ""
    signup_error = ""
    account_created = False
    if request.method == "POST":
        if 'login' in request.POST:
            username_or_email = request.POST.get("loginUsername")
            password = request.POST.get("loginPassword")
            try:
                user = UserAccount.objects.get(
                    models.Q(username=username_or_email) | models.Q(email=username_or_email),
                    password=password
                )
                request.session['user_id'] = user.id
                return redirect('home')
            except UserAccount.DoesNotExist:
                error = "Invalid username/email or password."
        elif 'signup' in request.POST:
            username = request.POST.get("signupUsername")
            email = request.POST.get("signupEmail")
            password = request.POST.get("signupPassword")
            confirm = request.POST.get("signupConfirm")
            if password != confirm:
                signup_error = "Passwords do not match."
            else:
                try:
                    UserAccount.objects.create(username=username, email=email, password=password)
                    account_created = True
                except IntegrityError:
                    signup_error = "Username or email already exists."
    return render(request, "log_sig.html", {
        "error": error,
        "signup_error": signup_error,
        "account_created": account_created
    })

def logout_view(request):
    request.session.flush()
    return redirect('home')
